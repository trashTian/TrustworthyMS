from featurize import smiles_to_data, collate_with_circle_index
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
import torch.distributed as dist
import os
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, matthews_corrcoef, f1_score
from model import *
from sklearn.metrics import average_precision_score
from featurize import smiles_to_data
import pandas as pd
from tqdm import tqdm
import torch
from Utils import beta_loss, custom_batching, NT_Xent, PairNorm, relu_evidence
import logging


def metric(label, probs):
    preds = list(map(lambda x: (x >= 0.5).astype(int), probs.detach().numpy()))
    auc = roc_auc_score(label, probs.detach().numpy())
    acc = accuracy_score(label, preds)
    precision = precision_score(label, preds)
    recall = recall_score(label, preds)
    f1 = f1_score(label, preds)
    mcc = matthews_corrcoef(label, preds)
    return auc, acc, precision, recall, f1, mcc


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='hold_out_classification.txt',  # 日志文件名
    filemode='a'  # 'w' 表示每次运行时覆盖文件，'a' 表示追加到文件
)


class MS_BACL(nn.Module):
    def __init__(self, num_features_xd=93, dropout_rate=0.5):
        super(MS_BACL, self).__init__()
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


def train_epoch(model, device, train_loader, optimizer, criterion):
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

        # 计算evidence
        evidence = relu_evidence(output)
        # 计算 alpha beta
        beta_alpha = evidence + 1
        # 计算不确定性
        u = (2 / torch.sum(beta_alpha, dim=1, keepdim=True)).view(-1)
        # 获取P [p_neg, p_pos]
        P = torch.softmax(output, dim=1)
        # 为正类的概率
        predict_prob = P[:, 1].view(-1, 1)

        # view1 和 view2 的损失
        loss_1, _ = criterion(output, target_one_hot)
        loss_2, _ = criterion(output1, target_one_hot)

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

            # 计算evidence
            evidence = relu_evidence(output)
            # 计算 alpha beta
            beta_alpha = evidence + 1
            # 计算不确定性
            u = (2 / torch.sum(beta_alpha, dim=1, keepdim=True)).view(-1)
            # 获取P [p_neg, p_pos]
            P = torch.softmax(output, dim=1)
            # 为正类的概率
            predict_prob = P[:, 1].view(-1, 1)

            preds.append(predict_prob.cpu())
            labels.append(data.y.cpu())

    return torch.cat(labels, dim=0), torch.cat(preds, dim=0)


def create_hold_out_data():
    train_list = pd.read_csv('data_External/train.csv')
    test_external_list = pd.read_csv('data_External/test.csv')
    test_rlm_list = pd.read_csv('data_rlm/test.csv')

    if not os.path.exists('data_External/train.pth'):
        DatasetTrain = []
        for idx, row in tqdm(train_list.iterrows()):
            data1 = smiles_to_data(row['SMILES'])
            data1.y = torch.tensor(row['Label'], dtype=torch.long)
            DatasetTrain.append(data1)
        torch.save(DatasetTrain, 'data_External/train.pth')

    if not os.path.exists('data_External/test_external.pth'):
        DatasetTest = []
        for idx, row in tqdm(test_external_list.iterrows()):
            data2 = smiles_to_data(row['SMILES'])
            data2.y = torch.tensor(row['Label'], dtype=torch.long)
            DatasetTest.append(data2)
        torch.save(DatasetTest, 'data_External/test.pth')

    if not os.path.exists('data_rlm/test_rlm.pth'):
        DatasetTest = []
        for idx, row in tqdm(test_rlm_list.iterrows()):
            data2 = smiles_to_data(row['SMILES'])
            data2.y = torch.tensor(row['Label'], dtype=torch.long)
            DatasetTest.append(data2)
        torch.save(DatasetTest, 'data_rlm/test_rlm.pth')


def train_predict():

    NUM_EPOCHS = 2000
    LR = 0.0005
    best_results_external = []
    best_results_rlm = []
    for i in range(1, 11):
        set_seed(i)
        test_external = []
        test_rlm = []

        data_train = torch.load("data_External/train.pth")
        data_test_external = torch.load('data_External/test.pth')
        data_test_rlm = torch.load('data_rlm/test_rlm.pth')

        batches_train = list(custom_batching(data_train, 256))
        batches_train1 = list()
        for batch_idx, data in enumerate(batches_train):
            data = collate_with_circle_index(data)
            data.edge_attr = None
            batches_train1.append(data)

        batches_test_external = list(custom_batching(data_test_external, 512))
        batches_test0 = list()
        for batch_idx, data in enumerate(batches_test_external):
            data = collate_with_circle_index(data)
            data.edge_attr = None
            batches_test0.append(data)

        batches_test_rlm = list(custom_batching(data_test_rlm, 512))
        batches_test1 = list()
        for batch_idx, data in enumerate(batches_test_rlm):
            data = collate_with_circle_index(data)
            data.edge_attr = None
            batches_test1.append(data)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = MS_BACL().cuda()
        criterion = beta_loss
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        max_external = 0
        max_rlm = 0
        for epoch in range(NUM_EPOCHS):
            train_labels, train_preds, lossz = train_epoch(model, device, batches_train1, optimizer, criterion)
            _, _, _, _, _, train_mcc = metric(train_labels, train_preds)

            # test on external
            test_labels0, test_preds0 = test_epoch(model, device, batches_test0)
            auc0, acc0, precision0, recall0, f1_scroe0, mcc0 = metric(test_labels0, test_preds0)
            test_external.append([auc0, acc0, f1_scroe0, mcc0])

            # test on rlm
            test_labels1, test_preds1 = test_epoch(model, device, batches_test1)
            auc1, acc1, precision0, recall1, f1_scroe1, mcc1 = metric(test_labels1, test_preds1)
            test_rlm.append([auc1, acc1, f1_scroe1, mcc1])

            if mcc0 > max_external:
                max_external = mcc0
                torch.save(model.state_dict(), "data_External/model_seed_{}.pth".format(i))
            if mcc1 > max_rlm:
                max_rlm = mcc1
                torch.save(model.state_dict(), "data_rlm/model_seed_{}.pth".format(i))

            logging.info(
                f'Epoch: {epoch} Loss: {lossz:.4f} Train MCC: {train_mcc:.4f} external MCC:{mcc0:.4f} rlm MCC:{mcc1:.4f}')

        data = np.array(test_external)
        best_epoch = np.argmax(data[:, 3])
        best_results_external.append(test_external[best_epoch])

        data = np.array(test_rlm)
        best_epoch = np.argmax(data[:, 3])
        best_results_rlm.append(test_rlm[best_epoch])

    logging.info('========result=====external=============')
    for re in best_results_external:
        logging.info(re)
    results_external = np.array(best_results_external)
    logging.info(np.mean(results_external, axis=0))
    logging.info(np.std(results_external, axis=0))

    logging.info('========result=====rlm=============')
    for re in best_results_rlm:
        logging.info(re)
    results_rlm = np.array(best_results_rlm)
    logging.info(np.mean(results_rlm, axis=0))
    logging.info(np.std(results_rlm, axis=0))


if __name__ == "__main__":
    train_predict()
