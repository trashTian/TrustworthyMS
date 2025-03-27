import numpy as np
from torch import nn
from featurize import smiles_to_data, collate_with_circle_index
import torch

from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn import GINConv
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, matthews_corrcoef, f1_score
from Utils import PairNorm, NT_Xent, beta_loss, custom_batching
import logging

# 配置 logging
logging.basicConfig(level=logging.INFO,  # 设置日志级别
                    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
                    handlers=[
                        logging.FileHandler("ablation_cls_wo_.log"),  # 日志文件
                        logging.StreamHandler()  # 控制台输出
                    ])


def metric(label, probs):
    preds = list(map(lambda x: (x >= 0.5).astype(int), probs.detach().numpy()))
    auc = roc_auc_score(label, probs.detach().numpy())
    acc = accuracy_score(label, preds)
    precision = precision_score(label, preds)
    recall = recall_score(label, preds)
    f1 = f1_score(label, preds)
    mcc = matthews_corrcoef(label, preds)
    return auc, acc, precision, recall, f1, mcc


class TrustworthyMS(nn.Module):
    def __init__(self, num_features_xd=93, dropout_rate=0.5):
        super(TrustworthyMS, self).__init__()
        self.fc_g_0 = nn.Sequential(
            nn.Linear(num_features_xd * 10 * 2, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512)
        )
        self.fc_g_1 = nn.Sequential(
            nn.Linear(43 * 10 * 2, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512)
        )
        self.fc_final_0 = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 2)
        )
        self.fc_final_1 = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 2)
        )
        self.conv1 = GINConv(nn.Linear(num_features_xd, num_features_xd))
        self.conv2 = GINConv(nn.Linear(num_features_xd, num_features_xd * 10))
        self.conv3 = GINConv(nn.Linear(43, 43))
        self.conv4 = GINConv(nn.Linear(43, 43 * 10))
        self.relu = nn.ReLU()
        self.norm = PairNorm(mode='None')

    def forward(self, x_0, edge_index_0, batch_0, x_1, edge_index_1, batch_1):
        x_g_0 = self.relu(self.conv1(x_0, edge_index_0))
        x_g_0 = self.relu(self.conv2(x_g_0, edge_index_0))
        x_g_0 = torch.cat([global_mean_pool(x_g_0, batch_0),
                           global_max_pool(x_g_0, batch_0)], dim=1)
        x_g_0 = self.norm(x_g_0)
        x_g_0 = self.fc_g_0(x_g_0)
        z = self.fc_final_0(x_g_0)

        x_g_1 = self.relu(self.conv3(x_1, edge_index_1))
        x_g_1 = self.relu(self.conv4(x_g_1, edge_index_1))
        x_g_1 = torch.cat([global_mean_pool(x_g_1, batch_1),
                           global_max_pool(x_g_1, batch_1)], dim=1)
        x_g_1 = self.norm(x_g_1)
        x_g_1 = self.fc_g_1(x_g_1)
        z1 = self.fc_final_1(x_g_1)

        return z, x_g_0, x_g_1, z1


def train_epoch(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    preds = []
    labels = []
    loss = 0

    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)

        optimizer.zero_grad()

        output, x_g, x_g1, output1 = model(
            data.x,
            data.edge_index,
            data.batch,
            data.x1.detach(),
            data.edge_index1.detach(),
            data.batch1.detach()
        )
        target_one_hot = torch.nn.functional.one_hot(data.y, num_classes=2).float()

        # 对比损失
        criterion_cl = NT_Xent(output.shape[0], 0.1, 1)
        cl_loss = criterion_cl(x_g, x_g1)

        P = torch.softmax(output, dim=1)
        # 为正类的概率
        predict_prob = P[:, 1].view(-1, 1)

        # view1 和 view2 的损失
        loss_1, _ = criterion(output=output,
                           target=target_one_hot,
                           device=device
                           )
        loss_2, _ = criterion(output=output1,
                           target=target_one_hot,
                           device=device)

        # 总损失
        loss = loss_1 + 0.3 * cl_loss + loss_2
        loss += loss.item()

        preds.append(predict_prob.cpu())
        labels.append(data.y.cpu())

        loss.backward()
        optimizer.step()

    return torch.cat(labels, dim=0), torch.cat(preds, dim=0), loss


def test_epoch(model, device, loader):
    model.eval()

    preds = []
    labels = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            output, x_g, x_g1, output1 = model(
                data.x,
                data.edge_index,
                data.batch,
                data.x1.detach(),
                data.edge_index1.detach(),
                data.batch1.detach()
            )
            P = torch.softmax(output, dim=1)
            # 为正类的概率
            predict_prob = P[:, 1].view(-1, 1)

            preds.append(predict_prob.cpu())
            labels.append(data.y.cpu())

    return torch.cat(labels, dim=0), torch.cat(preds, dim=0)


if __name__ == "__main__":
    NUM_EPOCHS = 300
    LR = 0.0001
    best_results = []
    for i in range(1, 11):
        test_AUC = []
        test_ACC = []
        test_F1 = []
        test_MCC = []

        data_train = torch.load(
            r"E:\ProgrammingSpace\Gitee\MSP\MSP\data\stratified_kfold_splits\train_fold_{}.pth".format(i))
        data_test = torch.load(
            r"E:\ProgrammingSpace\Gitee\MSP\MSP\data\stratified_kfold_splits\test_fold_{}.pth".format(i))

        batches_train = list(custom_batching(data_train, 256))
        batches_train1 = list()
        for batch_idx, data in enumerate(batches_train):
            data = collate_with_circle_index(data)
            data.edge_attr = None
            batches_train1.append(data)

        batches_test = list(custom_batching(data_test, 1024))
        batches_test1 = list()
        for batch_idx, data in enumerate(batches_test):
            data = collate_with_circle_index(data)
            data.edge_attr = None
            batches_test1.append(data)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = TrustworthyMS().cuda()
        criterion = beta_loss
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        for epoch in range(NUM_EPOCHS):
            train_labels, train_probs, lossz = train_epoch(model, device, batches_train1, optimizer, criterion, epoch)
            _, _, _, _, _, train_mcc = metric(train_labels, train_probs)
            test_labels, test_probs = test_epoch(model, device, batches_test1)
            auc, acc, precision, recall, f1, mcc = metric(test_labels, test_probs)
            logging.info(f'Epoch: {epoch} Loss: {lossz:.4f} Train MCC: {train_mcc:.4f} test MCC:{mcc:.4f}')
            test_AUC.append(auc)
            test_ACC.append(acc)
            test_F1.append(f1)
            test_MCC.append(mcc)

        best_results.append([max(test_AUC), max(test_ACC), max(test_F1), max(test_MCC)])
        logging.info('fold: {} {}'.format(i, str([max(test_AUC), max(test_ACC), max(test_F1), max(test_MCC)])))

    logging.info('========result=============')
    for re in best_results:
        logging.info(re)
    best_results = np.array(best_results)
    logging.info(np.mean(best_results, axis=0))
    logging.info(np.std(best_results, axis=0))
