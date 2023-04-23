import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression


def train_model(df, window_size):
    """
    训练模型
    :param window_size: int, 滑动窗口大小
    :return: LinearRegression模型和MinMaxScaler归一化器
    """



    # # 归一化输入输出数据
    # scaler = MinMaxScaler()
    # X_data = scaler.fit_transform(X_data)
    # y_data = scaler.fit_transform(y_data.reshape(-1, 1)).flatten()

    # 构建线性回归模型
    model = LinearRegression()
    model.fit(X_data, y_data)

    # return model, scaler
    return model


def predict(model, scaler=None, X=None):
    """
    使用模型进行预测

    :param model: LinearRegression模型
    :param scaler: MinMaxScaler归一化器
    :param data: numpy.ndarray, 一维时间序列数据
    :return: numpy.ndarray, 预测结果
    """
    # 归一化输入数据
    # X = scaler.transform(data[-model.window_size:].reshape(1, -1))

    # 使用模型进行预测
    y_pred = model.predict(X)

    # 反归一化预测结果
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    return y_pred

def create_dataset(df,window_size):
    # 构造输入输出数据
    X_data, y_data = [], []
    for i in range(window_size, len(df)):
        X_data.append(df.iloc[i - window_size:i, 0].values)
        y_data.append(df.iloc[i, 0])
    X_data = np.array(X_data)
    y_data = np.array(y_data)

# 测试代码
if __name__ == '__main__':
    n_step = 50
    data = pd.read_csv('../data/600000.SH.csv')
    train_data = data.iloc[:-n_step, -1:]
    test_data = data.iloc[-n_step:, -1:]

    # 训练模型
    # model, scaler = train_model(train_data, window_size=10)
    model = train_model(train_data, window_size=10)

    # 预测新数据
    # y_pred = predict(model, scaler, test_data)
    y_pred = predict(model, X=test_data)

    print(y_pred)
