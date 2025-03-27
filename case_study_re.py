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


class TrustworthyMS(nn.Module):
    def __init__(self, num_features_xd=93, dropout_rate=0.1):
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
        self.conv3 = GINConv(nn.Linear(43, 43))
        self.conv4 = GINConv(nn.Linear(43, 43 * 10))
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

        x_g_1 = self.relu(self.conv3(x_1, edge_index_1))
        x_g_1 = self.relu(self.conv4(x_g_1, edge_index_1))
        x_g_1 = torch.cat([global_mean_pool(x_g_1, batch_1),
                           global_max_pool(x_g_1, batch_1)], dim=1)
        x_g_1 = self.norm(x_g_1)
        x_g_1 = self.fc_g_1(x_g_1)
        z1 = self.fc_final_1(x_g_1)
        # alpha1 = torch.relu(z1[:, 0]) + 1
        # beta1 = torch.relu(z1[:, 1]) + 1
        alpha1 = z1[:, 0]
        beta1 = z1[:, 1]

        return alpha, beta, x_g_0, x_g_1, alpha1, beta1



def ten_fold_uncertainty(i=1):
    data_test = torch.load('case_re_data.pth')

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
        torch.load(r"E:\ProgrammingSpace\Gitee\MSP\MSP\data\regression_data\model_fold_{}.pth".format(i),
                   map_location=device)
    )
    model.eval()

    with torch.no_grad():
        for data in batches_test1:
            data = data.to(device)
            print(data[0])

            alpha, beta, x_g_0, x_g_1, alpha1, beta1 = model(
                data.x,
                data.edge_index,
                data.batch,
                data.x1.detach(),
                data.edge_index1.detach(),
                data.batch1.detach()
            )

            y_pred = alpha / (alpha + beta)

            u = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))

            print(y_pred)

            print(u)
            """
            tensor([0.8049, 0.3909], device='cuda:0')
            tensor([0.0798, 0.1431], device='cuda:0')
            """


ten_fold_uncertainty()
