import time
from keras.layers import LSTM, Dense, Bidirectional,Flatten
from keras.models import Sequential
from PyEMD.EMD import EMD


from utils.dataProcessing import *
from utils.plot import *
from utils.utils import *

"""利用emd-LSTM对数据进行单特征预测。(先emd分解在切分数据集)"""

"""
stepIn:相当于一次用多少个样本来预测，也就是滑动窗口的大小
stepOut: 相当于一次预测多少步y。目前的代码逻辑只能选择stepOut=1
"""


def trainAndTestBiLstm(deal_data_result, isShow=False):
    trainX, trainy, testX, testy, scaler = deal_data_result
    # 这里的trainy，testy可能是单维的，也可能是3维度的，需要reshape成2维的（方便训练和画图）
    trainy = trainy.reshape(trainy.shape[0], -1)
    testy = testy.reshape(testy.shape[0], -1)
    n_features = trainX.shape[2]
    model = Sequential()
    # params['stepIn']就是滑动窗口的长度（不算y）
    model.add(Bidirectional(
        LSTM(params['units'], activation='relu', input_shape=(params['stepIn'], n_features))
    ))
    model.add(Dense(trainy.shape[1]))
    model.compile(optimizer='adam', loss='mse')
    """ fit中加上：validation_data=(trainX, trainy)，不会让预测结果发生任何变化。相反还会降低训练的速度。
    所以如果不需要在训练过程中查看loss的变化，就不要加;  经过综合对比，batch_size = 64结合了准确率和速度，是最佳的选择。"""
    """
{'epoch': 20, 'units': 64, 'stepIn': 1, 'stepOut': 1, 'ration': 0.8}
RMSE: 59751.049	MAPE: 104.9774	PCC : 0.89	R2  : -12.28	用时：21.8s
{'epoch': 20, 'units': 64, 'stepIn': 1, 'stepOut': 1, 'ration': 0.8}
RMSE: 10271.8865	MAPE: 19.7295	PCC : 0.88	R2  : 0.61	用时：23.0s
{'epoch': 20, 'units': 64, 'stepIn': 1, 'stepOut': 1, 'ration': 0.8}
RMSE: 7997.3614	MAPE: 13.9007	PCC : 0.9	R2  : 0.76	用时：25.0s  128
{'epoch': 20, 'units': 64, 'stepIn': 1, 'stepOut': 1, 'ration': 0.8}
RMSE: 6530.5231	MAPE: 9.073	PCC : 0.92	R2  : 0.84	用时：28.4s
{'epoch': 20, 'units': 64, 'stepIn': 1, 'stepOut': 1, 'ration': 0.8}
RMSE: 6314.0755	MAPE: 8.8021	PCC : 0.92	R2  : 0.85	用时：34.4s  32

{'epoch': 20, 'units': 64, 'stepIn': 1, 'stepOut': 1, 'ration': 0.8}
RMSE: 6267.2559	MAPE: 8.8346	PCC : 0.92	R2  : 0.85	用时：48.1s  16
    """
    history = model.fit(trainX, trainy, epochs=params['epoch'], batch_size=64,
                        shuffle=True, verbose=0)
    # 展示训练过程中loss的变化
    showTrainLoss(history, isShow=isShow)
    # 预测
    # pred = model.predict(testX, verbose=0)
    X = np.concatenate((trainX, testX), axis=0)
    y = np.concatenate((trainy, testy), axis=0)
    pred = model.predict(X, verbose=0)

    # 反归一化
    pred = scaler.inverse_transform(pred)
    true = scaler.inverse_transform(y)
    # 展示预测结果和真实值
    # afterTrain(testy, pred, isShow)
    return true, pred


def trainAndTestMLP(deal_data_result, isShow=True):
    trainX, trainy, testX, testy, scalerX, scalery = deal_data_result
    # 这里的trainy，testy可能是单维的，也可能是3维度的，需要reshape成2维的（方便训练和画图）
    trainy = trainy.reshape(trainy.shape[0], -1)
    testy = testy.reshape(testy.shape[0],-1)
    n_features = trainX.shape[2]
    model = Sequential()
    # params['stepIn']就是滑动窗口的长度（不算y）
    model.add(Dense(paramsDLSTM['units'], activation='relu', input_shape=(paramsDLSTM['stepIn'], n_features),))
    model.add(Dense(paramsDLSTM['units'] // 2))
    model.add(Flatten())
    model.add(Dense(trainy.shape[1]))
    model.compile(optimizer='adam', loss='mse')
    """
    """  # validation_data=(trainX, trainy),
    history = model.fit(trainX, trainy, epochs=paramsDLSTM['epoch'], batch_size=50,validation_data=(trainX, trainy),
                        shuffle=True, verbose=0)
    # 展示训练过程中loss的变化
    showTrainLoss(history, isShow=isShow)
    # 预测
    pred = model.predict(testX, verbose=0)
    pred = pred.reshape((pred.shape[0],-1))
    # 反归一化
    pred = scalery.inverse_transform(pred)
    testy = testy.reshape((-1,1))
    testy = scalery.inverse_transform(testy)
    print(f'\n{paramsDLSTM}')
    afterTrain(testy, pred, isShow)
    saveTestResult(pred,'EMD-BiLSTM-MLP')
    return model



def main():
    init()
    isShow = False
    allData = getData('../data/agriculture_load_h.csv').values
    data = allData[:, -1]
    # print(f'\n{params}')
    # emd = EMD()
    # imfs = emd.emd(data)
    # yTrueList = []
    # yPredictList = []
    # for imf in imfs:
    #     """将每个分量依次送入LSTM中，并将LSTM计算的得到的预测值和真实值保持到列表中"""
    #     deal_data_result = deal_data(imf, params, mode='single')
    #     yTrue, yPredict = trainAndTestBiLstm(deal_data_result)
    #     yTrueList.append(yTrue)
    #     yPredictList.append(yPredict)
    # """对每个分量求和，得到总量（emd分解特点：序列分解后的imf分量之和，等于原始总量）"""
    # yTrue = np.sum(np.array(yTrueList), axis=0)
    # yPredict = np.sum(np.array(yPredictList), axis=0)
    # afterTrain(yTrue, yPredict, isShow)
    # # 保存Numpy对象到文件
    # np.save('result/yPredict.npy', yPredict)

    """第一阶段预测结束，第二阶段校正开始"""
    # 加载Numpy对象
    yPredict = np.load('result/yPredict.npy')
    # 由于滑动窗口的切分，导致预测出来的长度 = 原始长度 - stepIn - 1。为了能合并，需要对allData切分，使其长度和预测结果相同
    n = allData.shape[0] - params['stepIn'] - 1
    # 将 pred 合并到data 的倒数第二列,构造新数据集
    newData = np.concatenate([allData[-n:, :-1], yPredict, allData[-n:, -1:]], axis=1)
    deal_data_result = deal_data(newData, paramsDLSTM, mode='multiple')
    model = trainAndTestMLP(deal_data_result, isShow=False)



def afterTrain(yTrue, yPredict, isShow):
    # 展示预测结果和真实值
    if isShow:
        showTrueAndPredict1(yTrue, yPredict)
    # 计算各项评价指标
    result = evaluate(yTrue, yPredict)
    filename = os.path.basename(__file__).split('.')[0]
    saveResult(result, paramsDLSTM['stepIn'], f'{filename}.csv')


params = {'epoch': 20, 'units': 100, 'stepIn': 7, 'stepOut': 1, 'ration': 0.8}
paramsDLSTM = {'epoch': 40, 'units': 200, 'stepIn': 1, 'stepOut': 1, 'ration': 0.8}



if __name__ == "__main__":
    startTime = time.time()
    stepIn = 12
    for stepin in range(stepIn, stepIn + 1):
        # params['stepIn'] = stepin
        paramsDLSTM['stepIn'] = stepin
        main()
    timeCost = round(time.time() - startTime, 1)
    print(f'用时：{timeCost}s')


"""
{'epoch': 45, 'units': 200, 'stepIn': 27, 'stepOut': 1, 'ration': 0.8}
RMSE: 3545.25	MAPE: 4.22	PCC : 0.96	R2  : 0.93	用时：33.8s
"""