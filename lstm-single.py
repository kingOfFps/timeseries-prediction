import numpy as np
import joblib
import pandas as pd
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
import os
from utils import *
from dataProcessing import *

from sklearn.preprocessing import StandardScaler, MinMaxScaler


def train(train_data, stepIn, stepOut):
    """将普通的原始数据变为监督学习格式的数据"""
    train_X, train_y = prepare_y(train_data, stepIn, stepOut)
    # 当train_y是单维的时候，需要reshape成2维的
    train_y = train_y.reshape(train_y.shape[0], -1)
    n_features = train_X.shape[2]
    model = Sequential()
    model.add(LSTM(params['units'], activation='relu', input_shape=(stepIn, n_features)))
    """    # Dense层的参数也就是希望LSTM输出的数据的第二个维度的大小，也就是train_y.shape[1]。
    当train_y.shape[1] = 1,表示单步预测。当大于1，表示多步预测。也就是滑动窗口一次切分的数据，能预测处出步的标签y
    """
    model.add(Dense(train_y.shape[1]))
    model.compile(optimizer='adam', loss='mse')
    """画出训练过程的loss变化"""
    history = model.fit(train_X, train_y, validation_data=(train_X, train_y), epochs=params['epoch'], batch_size=64,
                        shuffle=True, verbose=0)
    # 展示训练过程
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()

    return model


# 模型预测函数
def predict(X, stepIn, stepOut):
    """测试数据X需要和训练数据做相同的滑动窗口切分处理，只不过训练集有标签y，真实场景下的预测没有y"""
    X, _ = prepare_data(X, np.array([]), stepIn, stepOut)
    pred = model.predict(X, verbose=0)
    return pred.reshape(pred.shape[0], -1)



"""配置各种参数
stepIn:相当于一次用多少个样本来预测，也就是滑动窗口的大小
stepOut: 相当于一次预测多少步y。目前的代码逻辑只能选择stepOut=1
"""
params = {'epoch': 30, 'units': 50, 'stepIn': 2, 'stepOut': 1, 'ration': 0.8}
if __name__ == "__main__":
    init()
    data = getData('load.csv')[['actual']]
    """切分数据集"""
    trainXy, testXy = splitTrainTest(data, params['ration'])
    # 归一化处理
    trainy, testy, scaler = spliteAndNormalizeY(trainXy, testXy)
    # 训练测试
    model = train(trainy, params['stepIn'], params['stepOut'])
    pred = predict(testy, params['stepIn'], params['stepOut'])
    testy = testy[:pred.shape[0], ]
    # 评估
    # evaluate(testy,pred)
    # 反归一化，使得数据变回原来的量纲
    pred = scaler.inverse_transform(pred)
    testy = scaler.inverse_transform(testy)

    """因为滑动窗口切分的原因，导致预测出来的pred大小和testData的大小不一样，
    所以这里进行处理，保证大小一样，方便后面画图。
    注意：这里切片用的是[-pred.shape[0]:, -1]，而不是[:pred.shape[0], -1]
    """
    # 画图对比测试集和预测结果
    showTruePred(testy, pred)
    print(params)
    evaluate(testy, pred)

"""
运行结果记录
{'epoch': 30, 'units': 50, 'stepIn': 2, 'stepOut': 1, 'ration': 0.8}
Test RMSE: 170.55
Test MAPE: 1.69
皮尔森系数0.99
决定系数0.97

{'epoch': 30, 'units': 50, 'stepIn': 3, 'stepOut': 1, 'ration': 0.8}
Test RMSE: 335.19
Test MAPE: 3.15
皮尔森系数0.94
决定系数0.88

{'epoch': 30, 'units': 50, 'stepIn': 4, 'stepOut': 1, 'ration': 0.8}
Test RMSE: 565.08
Test MAPE: 5.31
皮尔森系数0.82
决定系数0.66
"""
