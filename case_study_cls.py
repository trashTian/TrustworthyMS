# import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman']
# plt.rcParams.update({'font.size': 15})
# # 数据
# data = [6560, 6340, 940, 440, 70, 50, 0, 0]
# labels = ['Number {}'.format(i+1) for i in range(len(data))]  # 横轴标签
#
# # 创建图形
# fig, ax = plt.subplots()
#
# # 绘制从上到下的柱状图
# bars=ax.barh(labels, data, color='#09567a')
#
# # 设置横轴和左纵轴
# ax.spines['right'].set_visible(False)  # 隐藏右边框
# ax.spines['top'].set_visible(False)    # 隐藏上边框
#
# # 隐藏横轴刻度标签
# ax.set_yticks([])
#
# # 设置纵轴标签
# ax.set_xlabel('Number')  # 纵轴标签
#
# # 反转纵轴，使柱状图从上到下排列
# ax.invert_yaxis()
#
# xmax = max(data) * 1.1  # 横轴的最大值，留出一些空间
#
# # 在每个柱状图旁边显示数据值
# for bar in bars:
#     width = bar.get_width()  # 获取柱子宽度（即数据值）
#     if width > 0:  # 如果数据值为正数，则在柱子末端显示
#         ax.text(width + (xmax * 0.01),  # x坐标，稍微偏移避免覆盖柱子
#                 bar.get_y() + bar.get_height() / 2,  # y坐标，位于柱子中心
#                 f'{width}',  # 显示的文本
#                 va='center')  # 字体大小
#
#
# # 显示图表
# plt.show()


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

# 配置 logging
logging.basicConfig(level=logging.INFO,  # 设置日志级别
                    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
                    handlers=[
                        logging.FileHandler("temp1.log"),  # 日志文件
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
        self.norm = PairNorm()

    def forward(self, x_0, edge_index_0, batch_0, x_1, edge_index_1, batch_1):
        x_g_0 = self.relu(self.conv1(x_0, edge_index_0))
        x_g_0 = self.relu(self.conv2(x_g_0, edge_index_0))
        x_g_0 = torch.cat([global_max_pool(x_g_0, batch_0),
                           global_mean_pool(x_g_0, batch_0)], dim=1)
        x_g_0 = self.norm(x_g_0)
        x_g_0 = self.fc_g_0(x_g_0)
        z = self.fc_final_0(x_g_0)

        x_g_1 = self.relu(self.conv3(x_1, edge_index_1))
        x_g_1 = self.relu(self.conv4(x_g_1, edge_index_1))
        x_g_1 = torch.cat([global_max_pool(x_g_1, batch_1),
                           global_mean_pool(x_g_1, batch_1)], dim=1)
        x_g_1 = self.norm(x_g_1)
        x_g_1 = self.fc_g_1(x_g_1)
        z1 = self.fc_final_1(x_g_1)

        return z, x_g_0, x_g_1, z1


def acc_with_threshold(preds, labels, uncertainty, threshold):
    # Filter the data of an epoch according to threshold filter and return:
    # 1. the correct number of datapoints after filtering and 2. the number of filtered datapoints
    under_threshold_index = uncertainty < threshold
    preds_filter = preds[under_threshold_index]
    labels_filter = labels[under_threshold_index]
    filter_nums = len(labels_filter)
    match = torch.eq(preds_filter, labels_filter).float()
    acc_nums = torch.sum(match)
    return acc_nums, filter_nums


def filter_with_threshold(preds, labels, uncertainty, threshold):
    under_threshold_index = uncertainty < threshold
    preds_filter = preds[under_threshold_index]
    labels_filter = labels[under_threshold_index]
    return preds_filter, labels_filter


def ten_fold_uncertainty(i=1):
    data_test = torch.load('case_cls_data.pth')

    batches_test = list(custom_batching(data_test, 1024))
    batches_test1 = list()
    for batch_idx, data in enumerate(batches_test):
        data = collate_with_circle_index(data)
        data.edge_attr = None
        batches_test1.append(data)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TrustworthyMS()
    model = model.to(device)
    model.load_state_dict(
        torch.load(r"E:\ProgrammingSpace\Gitee\MSP\MSP\data\stratified_kfold_splits\test_fold_{}.pt".format(i),
                   map_location=device)
    )
    model.eval()

    with torch.no_grad():
        for data in batches_test1:
            data = data.to(device)
            print(data[0])
            output, x_g, x_g1, output1 = model(
                data.x,
                data.edge_index,
                data.batch,
                data.x1.detach(),
                data.edge_index1.detach(),
                data.batch1.detach()
            )

            _, preds = torch.max(output, 1)
            evidence = relu_evidence(output)
            beta_alpha = evidence + 1

            u = (2 / torch.sum(beta_alpha, dim=1, keepdim=True)).view(-1)
            print(preds)

            print(u)
            """
            tensor([1, 1], device='cuda:0')
            tensor([0.1076, 0.2536], device='cuda:0')
            """


ten_fold_uncertainty()

def create_hold_out_data():
    case_re = pd.read_csv(r'D:\case_study_reg.csv')
    case_cls = pd.read_csv(r'D:\case_study_cls.csv')

    if not os.path.exists('case_re_data.pth'):
        DatasetTrain = []
        for idx, row in tqdm(case_re.iterrows()):
            data1 = smiles_to_data(row['smiles'])
            data1.y = torch.tensor(row['logt'], dtype=torch.long)
            DatasetTrain.append(data1)
        torch.save(DatasetTrain, 'case_re_data.pth')

    if not os.path.exists('case_cls_data.pth'):
        DatasetTest = []
        for idx, row in tqdm(case_cls.iterrows()):
            data2 = smiles_to_data(row['SMILES'])
            data2.y = torch.tensor(row['Label'], dtype=torch.long)
            DatasetTest.append(data2)
        torch.save(DatasetTest, 'case_cls_data.pth')

# create_hold_out_data()