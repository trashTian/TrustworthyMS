import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
from scipy.stats import pearsonr


def metric(label, probs):

    mse = mean_squared_error(label, probs)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(label, probs)
    r2 = r2_score(label, probs)
    p, _ = pearsonr(label,probs)
    return rmse, mae, r2,p


values_true = []
for i in range(1, 11):
    df_true = pd.read_csv(r"E:\ProgrammingSpace\Gitee\MSP\MSP\data\regression_data\test_fold_{}.csv".format(i))
    value_true = df_true['logt'].values.tolist()
    values_true.append(value_true)

models = ['AttentiveFP', 'CMMS_GCL', 'D_MPNN', 'GAT', 'GBDT', 'MGCN', 'PredMS', 'XGBoost']

print('RMSE\tMAE\tR2\tp')
for model in models:
    df_pre = pd.read_csv(
        r"E:\ProgrammingSpace\Gitee\MSP\MSP\data\regression_data\{}_regression_results.csv".format(model))
    results = []
    for i in range(1, 11):
        fold_pre = df_pre['Fold_{}'.format(i)].dropna()
        value_pre = fold_pre.values.tolist()

        rmse, mae, r2 ,p= metric(values_true[i - 1], value_pre)

        results.append([rmse, mae, r2,p])
    print(model)
    res = np.array(results)
    print(np.mean(res, axis=0))
    print(np.std(res, axis=0))
