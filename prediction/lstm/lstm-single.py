import time
from keras.layers import LSTM, Dense
from keras.models import Sequential

from utils.dataProcessing import *
from utils.plot import *
from utils.utils import *

"""利用LSTM对数据进行单特征预测。"""

"""
stepIn:相当于一次用多少个样本来预测，也就是滑动窗口的大小
stepOut: 相当于一次预测多少步y。目前的代码逻辑只能选择stepOut=1
"""


def afterTrain(yTrue, yPredict, stepIn, isShow):
    # 展示预测结果和真实值
    if isShow:
        showTrueAndPredict1(yTrue, yPredict)
    # 计算各项评价指标
    result = evaluate(yTrue, yPredict)
    filename = os.path.basename(__file__).split('.')[0]
    saveResult(result, stepIn, f'{filename}.csv')

def trainAndTest(deal_data_result, isShow=True):
    trainX, trainy, testX, testy, scaler = deal_data_result
    # 这里的trainy，testy可能是单维的，也可能是3维度的，需要reshape成2维的（方便训练和画图）
    trainy = trainy.reshape(trainy.shape[0], -1)
    testy = testy.reshape(testy.shape[0], -1)
    n_features = trainX.shape[2]
    model = Sequential()
    # params['stepIn']就是滑动窗口的长度（不算y）
    model.add(LSTM(params['units'], activation='relu', input_shape=(params['stepIn'], n_features)))
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
    afterTrain(testy, pred, params['stepIn'], isShow)
    saveTestResult(pred,'LSTM')
    return model



def deal_data(data: np.ndarray):
    """
    0. 归一化（归一化放在预测之前是最科学的）
    1. 切分训练集、测试集
    2. 滑动窗口构造数据集
    :param data:从csv中读取的数据集
    """
    scaler = MinMaxScaler()
    data_scaler = scaler.fit_transform(data)
    X, y = prepare_y(data_scaler, params['stepIn'], params['stepOut'])
    # 切分数据集
    trainX, testX = splitTrainTest(X, params['ration'])
    trainy, testy = splitTrainTest(y, params['ration'])
    return trainX, trainy, testX, testy, scaler


def main():
    init()
    ration = 0.3
    data = getData('../data/agriculture_load_h.csv', ration).iloc[:, -1:]
    deal_data_result = deal_data(data.values)
    # 训练测试
    model = trainAndTest(deal_data_result, isShow=True)

params = {'epoch': 20, 'units': 64, 'stepIn': 1, 'stepOut': 1, 'ration': 0.8}

if __name__ == "__main__":
    startTime = time.time()
    for stepin in range(27, 28):
        params['stepIn'] = stepin
        print(f'\n{params}')
        main()
    timeCost = round(time.time() - startTime, 1)
    print(f'用时：{timeCost}s')

"""
ration:1,stepin:27,cost:22s,

1. 为什么"    attention_layer = Attention(use_scale=True)([lstm_layer, lstm_layer])"中传递了两个lstm_layer。
2. 如果不用Model构建神经网络模型，用Sequential的话，改如何实现。他们两者有在最后的效果上有什么区别。
3. 为什么用你的代码，预测出来的结果pred的维度和输入的训练标签testy对不上。比如我输入给神经网络训练的X的shape为(n,timestep,n_features),
训练的y的shape为(n,1)。但是预测出来的结果pred的shape为(n,timestep,n_features),而不是(n,1)，和用于训练的y一样大呢？
4.为什么把attention加到LSTM后面，可以加到其他地方嘛，相对于加到LSTM后面效果会怎么变化，为什么
5.我之前用Sequential我记得不需要特意加入Input层，第一层就是LSTM层。为什么现在用Model后，第一次需要先加上Input才能加LSTM。可以不加Input嘛
6.为什么return_sequences=True，use_scale=True。他们是干嘛的
7.attention能够加多层嘛，位置应当加在哪里才合适，Attention还有其他参数嘛。
8.Input层和Dense层，Flatten层有什么区别。进行时间序列预测的时候，为什么用MLP需要在最后的Dense层前加Flatten()层，LSTM不需要加。
"""
