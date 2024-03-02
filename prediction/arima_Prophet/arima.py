import time
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm

from utils.dataProcessing import *
from utils.plot import *
from utils.utils import *

params = {'ration': 0.8}


def afterTrain(yTrue, yPredict, isShow):
    # 展示预测结果和真实值
    if isShow:
        showTrueAndPredict1(yTrue, yPredict)
    # 计算各项评价指标
    result = evaluate(yTrue, yPredict)
    saveTestResult(yPredict, 'ARIMA')


def trainAndTest(data, p, q, isShow=True):
    # scaler1 = MinMaxScaler()
    # scaler2 = MinMaxScaler()
    trainy, testy = splitTrainTest(data, params['ration'])
    # trainy = scaler1.fit_transform(trainy)
    # scaler2 = scaler2.fit(testy)

    # 创建ARIMA模型并拟合数据
    model = ARIMA(trainy, order=(p, 1, q))
    model_fit = model.fit()
    # 预测
    pred = model_fit.forecast(steps=testy.shape[0])  # 修改steps以预测所需的步数
    # pred = scaler2.inverse_transform(pred.reshape((-1,1)))
    # 展示预测结果和真实值
    afterTrain(testy, pred, isShow)
    return model


def main():
    init()
    # 读取数据并解析日期
    data = getData('../data/agriculture_load_h.csv').iloc[:, -1:]
    # p, q = sm.tsa.arma_order_select_ic(data, max_ar=1, max_ma=1, ic='aic')['aic_min_order']
    order = sm.tsa.arma_order_select_ic(data, ic='aic', trend='n')
    p, q = order['aic_min_order']  # 获取选择的最佳阶数
    trainAndTest(data, p, q)


"""
RMSE: 16615.25	MAPE: 21.31	PCC : 0.05	R2  : -0.54  d=0
RMSE: 13245.02	MAPE: 20.07	PCC : 0.16	R2  : 0.02   d=1
RMSE: 14645.71	MAPE: 19.97	PCC : 0.21	R2  : -0.19

结论：best_order (p,d,q) = (4,1,2)
不加归一化效果好于加归一化
"""
if __name__ == '__main__':
    startTime = time.time()
    main()
    # print(params)
    # for stepin in range(1, 40):
    #     params['stepIn'] = stepin
    #     print(f'\n{params}')
    timeCost = round(time.time() - startTime, 1)
    print(f'用时：{timeCost}s')
