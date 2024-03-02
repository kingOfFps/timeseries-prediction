import time
from sklearn.model_selection import train_test_split
import pandas as pd
# from fbprophet import Prophet
from prophet import Prophet
import plotly.offline as py
from prophet.plot import add_changepoints_to_plot
import warnings
import logging

from utils.dataProcessing import *
from utils.plot import *
from utils.utils import *


params = {'ration': 0.8}
def trainAndTest(data,isShow=True):
    # 这里一定要设置shuffle=False
    trainy, testy = train_test_split(data,test_size=1-params['ration'],shuffle=False)
    # 初始化Prophet模型并训练
    # model = Prophet(daily_seasonality=True)
    model = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True)
    # model.fit(trainy, algorithm='LBFGS')
    model.fit(trainy)

    # 构建未来数据框架并进行预测
    future = model.make_future_dataframe(periods=testy.shape[0])
    forecast = model.predict(future)
    # 提取预测结果
    pred = forecast[['ds', 'yhat']].tail(testy.shape[0])
    # 展示预测结果和真实值
    afterTrain(testy, pred,  isShow)
    if isShow:
        showTrueAndPredict1(testy, pred)
    # 计算各项评价指标
    result = evaluate(testy, pred)
    return model



def main():
    init()
    logging.getLogger('fbprophet').setLevel(logging.WARNING)
    warnings.filterwarnings('ignore')
    plt.style.use('fivethirtyeight')
    # 读取数据并解析日期
    data = pd.read_csv('../data/agriculture_load_h.csv')
    # 把datetime列转换为日期格式
    data['datetime'] = pd.to_datetime(data['datetime'])
    # 把列名改为Prophet模型所需的ds和y
    data = data.rename(columns={'datetime': 'ds', 'load': 'y'})
    data = data[['ds','y']]
    trainAndTest(data)

    # evaluate(test_data['actual'], pred['yhat'])
    # showTruePred(test_data['actual'], pred['yhat'])

def afterTrain(yTrue, yPredict, isShow):
    # 展示预测结果和真实值
    if isShow:
        showTrueAndPredict1(yTrue, yPredict)
    # 计算各项评价指标
    result = evaluate(yTrue, yPredict)
    # filename = os.path.basename(__file__).split('.')[0]
    # saveResult(result, stepIn, f'{filename}.csv')

if __name__ == "__main__":
    startTime = time.time()
    main()
    print(params)
    timeCost = round(time.time() - startTime, 1)
    print(f'用时：{timeCost}s')


"""
14

100
Test RMSE: 1.38
Test MAPE: 8.11
皮尔森系数0.22
决定系数-41.05
150
Test RMSE: 0.46
Test MAPE: 3.09
皮尔森系数0.15
决定系数-3.74
200
Test RMSE: 0.4
Test MAPE: 2.46
皮尔森系数0.09
决定系数-2.48
300
Test RMSE: 0.54
Test MAPE: 3.96
皮尔森系数0.37
决定系数-5.36
"""