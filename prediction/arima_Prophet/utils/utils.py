import random
import matplotlib.pyplot as plt
# from keras.layers import GRU as LSTM
import numpy as np
import pandas as pd
import os
import csv


def init():
    """初始化，解决plt画图以及日志显示一系列问题"""
    import matplotlib as mpl
    import warnings
    import tensorflow as tf
    # 符号显示问题
    plt.rcParams['axes.unicode_minus'] = False
    # 汉字显示问题
    mpl.rcParams['font.family'] = 'SimHei'
    # 忽略不重要的日志
    warnings.filterwarnings('ignore')
    # 固定随机种子
    """设置随机种子，控制变量"""
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def mae(y_true, y_pred):
    """
    计算MAE（平均绝对误差）
    """
    return np.mean(np.abs(y_true - y_pred))


def mape(y_true, y_pred):
    """
    计算MAPE（平均绝对百分比误差）
    """
    return np.mean(np.abs((y_true - y_pred) / y_true + 0.00000001)) * 100


def mse(y_true, y_pred):
    """
    计算MSE（均方误差）
    """
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    """
    计算RMSE（均方根误差）
    """
    return np.sqrt(mse(y_true, y_pred))


def r2(y_true, y_pred):
    """
    计算R2（决定系数）
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 0.00000001
    return 1 - (ss_res / ss_tot)


def rmpe(y_true, y_pred):
    """
    计算RMPE（百分比均方根误差）
    """
    return np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2)) * 100


# 计算各种评价指标
def evaluate(yTrue, yPredict):
    yTrue = yTrue.reshape(-1)
    yPredict = yPredict.reshape(-1)
    from sklearn.metrics import mean_squared_error, r2_score
    from scipy.stats import pearsonr
    MSE = mean_squared_error(yTrue, yPredict)
    RMSE = round(np.sqrt(MSE), 2)
    MSE = round(MSE, 2)
    MAPE = round(mape(yTrue, yPredict), 2)
    pearsonrValue = round(pearsonr(yTrue, yPredict)[0], 2)
    r2 = round(r2_score(yTrue, yPredict), 2)

    print(f'RMSE: {RMSE}', end='\t')
    print(f'MAPE: {MAPE}', end='\t')
    print(f'PCC : {pearsonrValue}', end='\t')
    print(f'R2  : {r2}', end='\n')
    return {'RMSE': RMSE, 'MAPE': MAPE, 'PCC': pearsonrValue, 'R2': r2}


# 计算各种评价指标
def evaluate2(yTrue, yPredict):
    yTrue = np.array(yTrue)
    yPredict = np.array(yPredict)
    MAE = round(mae(yTrue, yPredict), 3)
    MAPE = round(mape(yTrue, yPredict), 2)
    MSE = round(mse(yTrue, yPredict), 2)
    RMPE = round(rmpe(yTrue, yPredict))
    RMSE = round(rmse(yTrue, yPredict), 3)
    R2 = round(r2(yTrue, yPredict) * 100, 2)
    result = {'MAE': MAE, 'MAPE': MAPE, 'MSE': MSE, 'RMPE': RMPE, 'RMSE': RMSE, 'R2': R2}
    print(result)
    return result


def saveResult(result, x, filename):
    filepath = f'result/{filename}'
    text = 'x,RMSE,MAPE,PCC,R2'
    # 如果指定的csv文件不存在，则创建。且写入指定的字符串
    if not os.path.exists(filepath):  # 检查文件是否存在
        with open(filepath, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([text])
    resultDf = pd.read_csv(filepath)
    # 创建新的一行数据
    new_row = {'x': x, 'RMSE': result['RMSE'], 'MAPE': result['MAPE'], 'PCC': result['PCC'], 'R2': result['R2']}
    # 将新行数据添加到DataFrame末尾
    # df = resultDf.append(new_row, ignore_index=True)
    df = pd.concat([resultDf, pd.DataFrame(new_row, index=[0])], ignore_index=True)
    # 将DataFrame保存回CSV文件
    # df.to_excel(f'result/{fileanme}', index=False)
    df.to_csv(filepath, index=False)


def saveTestResult(df, name):
    """保存每次模型预测的结果到csv文件中，方便展示.
    因为不同步长得到的预测结果长度不相同，需要切割成统一长度的来对齐每个时间点。
    目前520是相对ration = 0.3来说，能保证步长在40内的最大切割长度了。"""
    count = 520
    # data = pd.read_csv('../../data/result.csv', encoding='utf8')
    filepath = "../lstm/result/summaryOfResults.xlsx"
    data = pd.read_excel(filepath)
    # 将每列的数据向前移动一行
    df = pd.DataFrame(df).iloc[-count:, ]
    df = df.reset_index(drop=True)
    df.columns = [name]
    # df = df[:data.shape[0]]
    if name in data.columns:
        data = data.drop([name], axis=1)
    # data = data.apply(lambda x: x.fillna(x.shift()))
    data = pd.concat([data, df], axis=1)
    # data = data.rename(columns={0: name})
    data.to_excel(filepath, index=False)


if __name__ == '__main__':
    def example_generator(in_list):
        '''生成器'''
        for i in in_list:
            yield i * 2


    r = example_generator([1, 2, 3])
