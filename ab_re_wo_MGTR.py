from scipy.stats import pearsonr

from featurize import smiles_to_data, collate_with_circle_index
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
import torch.distributed as dist
import os
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from model import *
from sklearn.metrics import average_precision_score
from featurize import smiles_to_data
import pandas as pd
from tqdm import tqdm
import torch
from Utils import beta_loss, custom_batching, NT_Xent, PairNorm, relu_evidence
import logging


def metric(label, probs):
    probs = probs.detach().numpy()
    mse = mean_squared_error(label, probs)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(label, probs)
    r2 = r2_score(label, probs)
    p, _ = pearsonr(label, probs)
    return rmse, mae, r2, p


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='ablation_reg_wo_MGTR.txt',  # 日志文件名
    filemode='a'  # 'w' 表示每次运行时覆盖文件，'a' 表示追加到文件
)


class MS_BACL(nn.Module):
    def __init__(self, num_features_xd=93, dropout_rate=0.1):
        super(MS_BACL, self).__init__()
        self.fc_g_0 = nn.Sequential(
            nn.Linear(num_features_xd * 10 * 2, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512)
        )
        self.fc_g_1 = nn.Sequential(
            nn.Linear(num_features_xd * 10 * 2, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512)
        )
        self.fc_final_0 = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 2),
            nn.Sigmoid()
        )
        self.fc_final_1 = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 2),
            nn.Sigmoid()
        )
        self.conv1 = GINConv(nn.Linear(num_features_xd, num_features_xd))
        self.conv2 = GINConv(nn.Linear(num_features_xd, num_features_xd * 10))
        self.conv3 = GINConv(nn.Linear(num_features_xd, num_features_xd))
        self.conv4 = GINConv(nn.Linear(num_features_xd, num_features_xd * 10))
        self.relu = nn.ReLU()
        self.norm = PairNorm()

    def forward(self, x_0, edge_index_0, batch_0, x_1, edge_index_1, batch_1):
        x_g_0 = self.relu(self.conv1(x_0, edge_index_0))
        x_g_0 = self.relu(self.conv2(x_g_0, edge_index_0))
        x_g_0 = torch.cat([global_mean_pool(x_g_0, batch_0),
                           global_max_pool(x_g_0, batch_0)], dim=1)
        x_g_0 = self.norm(x_g_0)
        x_g_0 = self.fc_g_0(x_g_0)
        z = self.fc_final_0(x_g_0)
        # alpha = torch.relu(z[:, 0]) + 1
        # beta = torch.relu(z[:, 1]) + 1
        alpha = z[:, 0]
        beta = z[:, 1]

        x_g_1 = self.relu(self.conv3(x_0, edge_index_0))
        x_g_1 = self.relu(self.conv4(x_g_1, edge_index_0))
        x_g_1 = torch.cat([global_mean_pool(x_g_1, batch_0),
                           global_max_pool(x_g_1, batch_0)], dim=1)
        x_g_1 = self.norm(x_g_1)
        x_g_1 = self.fc_g_1(x_g_1)
        z1 = self.fc_final_1(x_g_1)
        # alpha1 = torch.relu(z1[:, 0]) + 1
        # beta1 = torch.relu(z1[:, 1]) + 1
        alpha1 = z1[:, 0]
        beta1 = z1[:, 1]

        return alpha, beta, x_g_0, x_g_1, alpha1, beta1


def train_epoch(model, device, train_loader, optimizer, criterion):
    model.train()
    preds = []
    labels = []
    loss = 0

    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)

        optimizer.zero_grad()

        alpha, beta, x_g_0, x_g_1, alpha1, beta1 = model(
            data.x,
            data.edge_index,
            data.batch,
            data.x1.detach(),
            data.edge_index1.detach(),
            data.batch1.detach()
        )

        # 对比损失
        criterion_cl = NT_Xent(alpha.shape[0], 0.1, 1)
        cl_loss = criterion_cl(x_g_0, x_g_1)

        # view1 和 view2 的损失
        y_pred = alpha / (alpha + beta)
        y_pred1 = alpha1 / (alpha1 + beta1)
        loss_1 = criterion(y_pred, data.y)
        loss_2 = criterion(y_pred1, data.y)

        # 总损失
        loss = loss_1 + 0.3 * cl_loss + loss_2
        loss += loss.item()

        preds.append(y_pred.cpu())
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

            alpha, beta, x_g_0, x_g_1, alpha1, beta1 = model(
                data.x,
                data.edge_index,
                data.batch,
                data.x1.detach(),
                data.edge_index1.detach(),
                data.batch1.detach()
            )

            y_pred = alpha / (alpha + beta)

            preds.append(y_pred.cpu())
            labels.append(data.y.cpu())

    return torch.cat(labels, dim=0), torch.cat(preds, dim=0)


def create_hold_out_data():
    for i in range(1, 11):
        train_list = pd.read_csv(r'E:\ProgrammingSpace\Gitee\MSP\MSP\data\regression_data\train_fold_{}.csv'.format(i))
        test_list = pd.read_csv(r'E:\ProgrammingSpace\Gitee\MSP\MSP\data\regression_data\test_fold_{}.csv'.format(i))

        if not os.path.exists(r'E:\ProgrammingSpace\Gitee\MSP\MSP\data\regression_data\train_fold_{}.pth'.format(i)):
            DatasetTrain = []
            for idx, row in tqdm(train_list.iterrows()):
                data1 = smiles_to_data(row['smiles'])
                data1.y = torch.tensor(row['logt'], dtype=torch.float)
                DatasetTrain.append(data1)
            torch.save(DatasetTrain,
                       r'E:\ProgrammingSpace\Gitee\MSP\MSP\data\regression_data\train_fold_{}.pth'.format(i))

        if not os.path.exists(r'E:\ProgrammingSpace\Gitee\MSP\MSP\data\regression_data\test_fold_{}.pth'.format(i)):
            DatasetTest = []
            for idx, row in tqdm(test_list.iterrows()):
                data2 = smiles_to_data(row['smiles'])
                data2.y = torch.tensor(row['logt'], dtype=torch.float)
                DatasetTest.append(data2)
            torch.save(DatasetTest,
                       r'E:\ProgrammingSpace\Gitee\MSP\MSP\data\regression_data\test_fold_{}.pth'.format(i))


def train_predict():
    NUM_EPOCHS = 200
    LR = 0.0005
    best_results = []
    for i in range(1, 11):

        data_train = torch.load(r"E:\ProgrammingSpace\Gitee\MSP\MSP\data\regression_data\train_fold_{}.pth".format(i))
        data_test = torch.load(r'E:\ProgrammingSpace\Gitee\MSP\MSP\data\regression_data\test_fold_{}.pth'.format(i))

        batches_train = list(custom_batching(data_train, 256))
        batches_train1 = list()
        for batch_idx, data in enumerate(batches_train):
            data = collate_with_circle_index(data)
            data.edge_attr = None
            batches_train1.append(data)

        batches_test_external = list(custom_batching(data_test, 512))
        batches_test0 = list()
        for batch_idx, data in enumerate(batches_test_external):
            data = collate_with_circle_index(data)
            data.edge_attr = None
            batches_test0.append(data)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = MS_BACL().cuda()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        max_r2 = 0
        max_res = []
        for epoch in range(NUM_EPOCHS):
            train_labels, train_preds, lossz = train_epoch(model, device, batches_train1, optimizer, criterion)
            train_rmse, train_mae, train_r2, train_p = metric(train_labels, train_preds)

            # test on external
            test_labels0, test_preds0 = test_epoch(model, device, batches_test0)
            test_rmse, test_mae, test_r2, test_p = metric(test_labels0, test_preds0)

            if test_r2 > max_r2:
                max_r2 = test_r2
                # torch.save(model.state_dict(),
                #            r"E:\ProgrammingSpace\Gitee\MSP\MSP\data\regression_data\model_fold_{}.pth".format(i))
                max_res = [test_r2, test_mae, test_rmse, test_p]

            logging.info(f'Epoch: {epoch} '
                         f'Loss {lossz:.4f} '
                         f'Train RMSE {train_rmse:.4f}, MAE {train_mae:.4f}, R2 {train_r2:.4f}, P {train_p:.4f} '
                         f'Test RMSE {test_rmse:.4f}, MAE {test_mae:.4f}, R2 {test_r2:.4f}, P {test_p:.4f} ')
        best_results.append(max_res)

    for re in best_results:
        logging.info(re)

    results = np.array(best_results)
    logging.info(np.mean(results, axis=0))
    logging.info(np.std(results, axis=0))


if __name__ == "__main__":
    # create_hold_out_data()
    train_predict()
