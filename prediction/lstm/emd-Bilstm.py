import time
from keras.layers import LSTM, Dense, Bidirectional
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


def trainAndTest(deal_data_result, isShow=False):
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
    """画出训练过程的loss变化"""
    history = model.fit(trainX, trainy, validation_data=(trainX, trainy), epochs=params['epoch'], batch_size=64,
                        shuffle=True, verbose=0)
    # 展示训练过程中loss的变化
    showTrainLoss(history, isShow=isShow)
    # 预测
    pred = model.predict(testX, verbose=0)
    # 反归一化
    pred = scaler.inverse_transform(pred)
    testy = scaler.inverse_transform(testy)
    # 展示预测结果和真实值
    # afterTrain(testy, pred, params['stepIn'], isShow)
    return testy, pred


def afterTrain(yTrue, yPredict, stepIn, isShow):
    # 展示预测结果和真实值
    if isShow:
        showTrueAndPredict1(yTrue, yPredict)
    # 计算各项评价指标
    result = evaluate(yTrue, yPredict)
    filename = os.path.basename(__file__).split('.')[0]
    saveResult(result, stepIn, f'{filename}.csv')


def deal_data(data: np.ndarray):
    """
    0. 归一化（归一化放在预测之前是最科学的）
    1. 切分训练集、测试集
    2. 滑动窗口构造数据集
    :param data:从csv中读取的数据集
    """
    scaler = MinMaxScaler()
    # 这里data为imf，为一维的，需要reshape为二维的
    data_scaler = scaler.fit_transform(data.reshape((-1, 1)))
    X, y = prepare_y(data_scaler, params['stepIn'], params['stepOut'])
    # 切分数据集
    trainX, testX = splitTrainTest(X, params['ration'])
    trainy, testy = splitTrainTest(y, params['ration'])
    return trainX, trainy, testX, testy, scaler


def main():
    init()
    isShow = False
    ration = 0.3
    data = getData('../data/agriculture_load_h.csv', ration).iloc[:, -1].values
    emd = EMD()
    imfs = emd.emd(data)
    yTrueList = []
    yPredictList = []
    for imf in imfs:
        """将每个分量依次送入LSTM中，并将LSTM计算的得到的预测值和真实值保持到列表中"""
        deal_data_result = deal_data(imf)
        yTrue, yPredict = trainAndTest(deal_data_result)
        yTrueList.append(yTrue)
        yPredictList.append(yPredict)
    """对每个分量求和，得到总量（emd分解特点：序列分解后的imf分量之和，等于原始总量）"""
    yTrue = np.sum(np.array(yTrueList), axis=0)
    yPredict = np.sum(np.array(yPredictList), axis=0)
    # 展示预测结果和真实值
    afterTrain(yTrue, yPredict, params['stepIn'], isShow)
    saveTestResult(yPredict, 'EMD-BiLSTM')


params = {'epoch': 20, 'units': 64, 'stepIn': 1, 'stepOut': 1, 'ration': 0.8}

if __name__ == "__main__":
    startTime = time.time()
    stepIn = 8
    for stepin in range(stepIn, stepIn + 1):
        params['stepIn'] = stepin
        print(f'\n{params}')
        main()
    timeCost = round(time.time() - startTime, 1)
    print(f'用时：{timeCost}s')

"""
本机：
stepIn:1~30;ration:0.2;cost:1163;
stepIn:1~30;ration:0.5;cost:3000;
stepIn:28;ration:1;cost:276s;
stepIn:28;ration:0.5;cost:120s;
stepIn:28;ration:0.2;cost:45s;
服务器：
stepIn:1~30;ration:0.2;cost:4516; 
"""
