import random
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


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
    return np.mean(np.abs((y_true - y_pred) / y_true+0.00000001)) * 100


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
    MSE = round(MSE, 4)
    MAPE = round(mape(yTrue, yPredict), 2)
    pearsonrValue = round(pearsonr(yTrue, yPredict)[0], 2)
    r2 = round(r2_score(yTrue, yPredict), 2)

    print(f'Test RMSE: {RMSE}')
    # print(f'Test RMSE: {rmse}')
    print(f'Test MAPE: {MAPE}')
    print(f'皮尔森系数{pearsonrValue}')
    print(f'决定系数{r2}')
    return {'RMSE': rmse, 'MAPE': mape, 'Person': pearsonrValue, 'R2': r2}


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


def showTruePred(yTrue, yPredict):
    plt.plot(yTrue, label='yTrue')
    plt.plot(yPredict, label='yPred')
    plt.legend()
    plt.show()
