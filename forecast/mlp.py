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


def train(trainX, trainy, stepIn, stepOut):
    """id为要训练设备的id（电表、水表、天然气设备的id），这里需要id，是因为train需要根据id来进行训练模型的命名"""
    """将普通的原始数据变为监督学习格式的数据"""
    trainX, trainy = prepare_data(trainX, trainy, stepIn, stepOut)
    # 当trainy是单维的时候，需要reshape成2维的
    trainy = trainy.reshape(trainy.shape[0], -1)
    n_features = trainX.shape[2]
    model = Sequential()
    # model.add(Dense(params['units'], activation='relu', input_shape=(stepIn, n_features)))
    model.add(Dense(params['units'], activation='relu', input_shape=(trainX.shape[1], trainX.shape[2])))
    # model.add(Dense(params['units']//2, activation='relu'))
    model.add(Dense(trainy.shape[1]))
    model.compile(optimizer='adam', loss='mse')
    """画出训练过程的loss变化"""
    history = model.fit(trainX, trainy, validation_data=(trainX, trainy), epochs=params['epoch'], batch_size=64,
                        shuffle=True,verbose=0)
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
def predict(X, model,stepIn,stepOut):
    scalerX = MinMaxScaler()
    X = scalerX.fit_transform(X)
    """测试数据X需要和训练数据做相同的滑动窗口切分处理，只不过训练集有标签y，真实场景下的预测没有y"""
    X, _ = prepare_data(X, np.array([]), stepIn, stepOut)
    pred = model.predict(X, verbose=0)
    return pred[:,0,:]
    # return pred.reshape(pred.shape[0], -1)


"""配置各种参数
stepIn:相当于一次用多少个样本来预测，也就是滑动窗口的大小
stepOut: 相当于一次预测多少步y.目前的代码逻辑只能选择stepOut=1
"""
params = {'epoch': 30, 'units': 64,'stepIn':4,'stepOut':1,'ration':0.8}
if __name__ == "__main__":
    init()
    data = getData('../data/baNaMa.xlsx')
    """切分数据集"""
    trainXy, testXy = splitTrainTest(data, params['ration'])
    # 归一化处理
    trainX, trainy, testX, testy, scalerX, scalery = spliteAndNormalizeXy(trainXy, testXy)
    # 训练测试
    model = train(trainX, trainy, params['stepIn'], params['stepOut'])
    pred = predict(testX, model,params['stepIn'], params['stepOut'])
    testy = testy[-pred.shape[0]:,]
    # 评估
    # evaluate(testy,pred)
    pred = scalery.inverse_transform(pred)
    testy = scalery.inverse_transform(testy)

    """因为滑动窗口切分的原因，导致预测出来的pred大小和testData的大小不一样，
    所以这里进行处理，保证大小一样，方便后面画图。
    注意：这里切片用的是[-pred.shape[0]:, -1]，而不是[:pred.shape[0], -1]
    """
    # pred = pred[:, -1]
    # 画图对比测试集和预测结果
    showTruePred(testy,pred)
    print(params)
    result = evaluate(testy,pred)


"""
记录训练结果：
{'epoch': 30, 'units': 64, 'stepIn': 1, 'stepOut': 1, 'ration': 0.8}
Test RMSE: 143.04
Test MAPE: 10.82
皮尔森系数0.83
决定系数0.46
{'epoch': 30, 'units': 64, 'stepIn': 2, 'stepOut': 1, 'ration': 0.8}
Test RMSE: 163.32
Test MAPE: 10.32
皮尔森系数0.73
决定系数0.3


"""