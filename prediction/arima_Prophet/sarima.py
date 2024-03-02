import time
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import pmdarima as pm

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
    trainy, testy = splitTrainTest(data, params['ration'])
    # 自动确定 SARIMA 模型的超参数
    # order = pm.auto_arima(data, seasonal=True, m=12)
    # print(order)
    # model = sm.tsa.statespace.SARIMAX(trainy, order=order.order,seasonal_order=order.seasonal_order)
    model = sm.tsa.statespace.SARIMAX(trainy, order=(4, 1, 2),
                                      seasonal_order=(2, 0, 1, 12))
    model_fit = model.fit(disp=-1)
    # 预测
    pred = model_fit.forecast(steps=testy.shape[0])  # 修改steps以预测所需的步数
    # 展示预测结果和真实值
    afterTrain(testy, pred, isShow)

    return model


def main():
    init()
    # 读取数据并解析日期
    data = getData('../data/agriculture_load_h.csv').iloc[:, -1:]
    # p, q = sm.tsa.arma_order_select_ic(data, max_ar=1, max_ma=1, ic='aic')['aic_min_order']
    order = sm.tsa.arma_order_select_ic(data, ic='aic', trend='n')
    print(order)
    p, q = order['aic_min_order']  # 获取选择的最佳阶数
    trainAndTest(data, p, q)


"""
 ARIMA(1,1,1)(2,0,1)[12]          
RMSE: 13543.63	MAPE: 19.8	PCC : 0.17	R2  : -0.02
 ARIMA(4,1,2)(2,0,1)[12]     best      
RMSE: 13230.62	MAPE: 20.11	PCC : 0.25	R2  : 0.03

结论：best_order (p,d,q) = 
不加归一化效果好于加归一化


"""
if __name__ == '__main__':
    startTime = time.time()
    main()
    timeCost = round(time.time() - startTime, 1)
    print(f'用时：{timeCost}s')
