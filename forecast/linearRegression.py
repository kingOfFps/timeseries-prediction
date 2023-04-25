import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from forecast.utils import *
import joblib
import os


def train_model(df, window_size):
    """
    训练模型
    :param window_size: int, 滑动窗口大小
    :return: LinearRegression模型和MinMaxScaler归一化器
    """

    # X_data, y_data = train_data.iloc[:, :-1], train_data.iloc[:, -1:]
    X_data, y_data = create_dataset(df, window_size)
    # # 归一化输入输出数据
    # scaler = MinMaxScaler()
    # X_data = scaler.fit_transform(X_data)
    # y_data = scaler.fit_transform(y_data.reshape(-1, 1)).flatten()

    # 构建线性回归模型
    model = LinearRegression()
    model.fit(X_data, y_data)
    # 保存模型
    path = f'forecast/model'
    if not os.path.exists(path):
        path = "./model"
    joblib.dump(model, f'{path}/linearRegression.pkl')
    # return model, scaler
    return model


def predict(model, scaler=None, X=None, window_size=1, n_step=7):
    """
    使用模型进行预测

    :param model: LinearRegression模型
    :param scaler: MinMaxScaler归一化器
    :param data: numpy.ndarray, 一维时间序列数据
    :return: numpy.ndarray, 预测结果
    """
    # 归一化输入数据
    # X = scaler.transform(data[-model.window_size:].reshape(1, -1))

    # X = create_X(X,window_size)

    y_pred = []
    for i in range(n_step):
        X = X.reshape((1,-1))
        result = model.predict(X)
        y_pred.append(result)
        # 删除第一个元素
        X = np.delete(X, 0)
        # 在最后面添加一个元素
        X = np.append(X, result)

    # 使用模型进行预测
    # y_pred = model.predict(X)

    # 反归一化预测结果
    # y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    return y_pred


def create_dataset(df, window_size):
    # 构造输入输出数据
    X_data, y_data = [], []
    for i in range(window_size, len(df)):
        X_data.append(df.iloc[i - window_size:i, 0].values)
        y_data.append(df.iloc[i, 0])
    X_data = np.array(X_data)
    y_data = np.array(y_data)
    return X_data, y_data


def create_X(df, window_size):
    # 构造输入输出数据
    X_data = []
    for i in range(window_size, len(df)):
        X_data.append(df.iloc[i - window_size:i, 0].values)
    X_data = np.array(X_data)
    return X_data


def fun():
    n_step = 50
    data = pd.read_csv('../data/600000.SH.csv')
    train_data = data[['close']][:-n_step]
    test_data = data[['close']][-n_step:]
    window_size = 5
    # 训练模型
    # model, scaler = train_model(train_data, window_size=10)
    model = train_model(train_data, window_size=window_size)

    # 预测新数据
    # y_pred = predict(model, scaler, test_data)
    X = train_data.values[-window_size:, ].reshape((1,-1))
    pred = predict(model, X=X, window_size=window_size, n_step=n_step)
    test_data = test_data.values.reshape((-1))
    # test_data = test_data[:-1]
    # pred = pred[1:]
    evaluate(test_data, pred)
    showTruePred(test_data, pred)


# 测试代码
if __name__ == '__main__':
    fun()
    # n_step = 50
    # data = pd.read_csv('../data/600000.SH.csv')
    # train_data = data[['close']][:-n_step]
    # test_data = data[['close']][-n_step:]
    # window_size = 7
    # # 训练模型
    # # model, scaler = train_model(train_data, window_size=10)
    # model = train_model(train_data, window_size=window_size)
    #
    # # 预测新数据
    # # y_pred = predict(model, scaler, test_data)
    # pred = predict(model, X=test_data,window_size= window_size)
    # test_data = test_data.iloc[-pred.shape[0]:,]
    # test_data = test_data.values.reshape((-1))
    #
    # test_data = test_data[:-1]
    # pred = pred[1:]
    # evaluate(test_data, pred)
    # showTruePred(test_data, pred)

"""
Test RMSE: 0.12
Test MAPE: 1.01
皮尔森系数0.76
决定系数0.28

Test RMSE: 0.02
Test MAPE: 0.14
皮尔森系数1.0
决定系数1.0
"""
