from featurize import smiles_to_data
import pandas as pd
from tqdm import tqdm
import torch
import os


root = 'data'
list1 = pd.read_csv(root + '/train.csv')
listtest = pd.read_csv(root + '/test.csv')


if not os.path.exists(root + '/train.pth'):
    DatasetTrain = []
    for idx, row in tqdm(list1.iterrows()):
        data1 = smiles_to_data(row['SMILES'])
        data1.y = torch.tensor(row['Label'], dtype=torch.long)
        DatasetTrain.append(data1)
    torch.save(DatasetTrain, root + '/train.pth')

if not os.path.exists(root + '/train.pth'):
    DatasetTest = []
    for idx, row in tqdm(listtest.iterrows()):
        data2 = smiles_to_data(row['SMILES'])
        data2.y = torch.tensor(row['Label'], dtype=torch.long)
        DatasetTest.append(data2)
    torch.save(DatasetTest, root + '/train.pth')


