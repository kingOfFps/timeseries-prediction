import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from forecast.utils import *
from keras.layers import Bidirectional


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        dataX.append(dataset[i:i + look_back, :])
        dataY.append(dataset[i + look_back, -1])
    return np.array(dataX), np.array(dataY)





def forecast(data, n_step):
    data['trade_date'] = pd.to_datetime(data['trade_date'], format='%Y%m%d')
    data.set_index('trade_date', inplace=True)
    data.sort_index(inplace=True)

    # 使用开盘价、最高价、最低价和收盘价
    features = data[['open', 'high', 'low', 'pre_close','change','pct_chg','vol','amount', 'close']].values

    X, y = features[:, :-1], features[:, -1:]
    # 数据归一化
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    features_scaled = np.concatenate((X_scaled, y_scaled), axis=1)

    # 构建数据集
    look_back = 4
    X_train, y_train = create_dataset(features_scaled, look_back)

    # 构建LSTM模型
    model = Sequential()
    model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])))
    # model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # 训练模型
    history = model.fit(X_train, y_train, epochs=20, batch_size=512, verbose=0, validation_data=(X_train, y_train))
    # showTrain(history)
    print('训练完毕')

    # 预测
    # input_data = features_scaled[-look_back:]
    input_data = X_train[-1]
    input_data = input_data.reshape((1, input_data.shape[0], input_data.shape[1]))
    predictions = predict(model, input_data, n_step)
    # 反归一化预测结果
    close_column_index = -1
    predictions = np.array(predictions).reshape(-1, 1)
    # predictions = scaler.inverse_transform(np.insert(predictions, [0] * (data.shape[1]-1), 0, axis=1))[:, close_column_index]
    predictions = scaler_y.inverse_transform(predictions)
    print('预测完毕')
    return predictions


def predict(model, input_data, step):
    predictions = []
    for _ in range(step):
        prediction = model.predict(input_data)
        predictions.append(prediction[0][0])

        new_input_data = np.roll(input_data, -1)
        new_input_data[0, 0, -1] = prediction
        input_data = new_input_data
    return predictions




if __name__ == "__main__":
    n_step = 10
    count = 30
    data = pd.read_csv('../data/000001.SZ.csv').iloc[:count,:]
    # 将data反转，使其按照时间顺序排列
    data = data.iloc[::-1]
    # plt.plot(data.loc[:200,['close']])
    # plt.show()

    train_data = data.iloc[:-n_step, :]
    test_data = data.iloc[-n_step:, :]
    pred = forecast(train_data, n_step)
    print(count)
    evaluate(test_data['close'], pred)
    showTruePred(test_data['close'], pred)

"""
n_step=14
300
Test RMSE: 0.27
Test MAPE: 1.77
皮尔森系数0.44
决定系数-0.58




"""
