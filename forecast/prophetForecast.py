import pandas as pd
# from fbprophet import Prophet
from prophet import Prophet
from forecast.utils import *

def forecast(data, steps_to_predict):
    data['trade_date'] = pd.to_datetime(data['trade_date'], format='%Y%m%d')
    data = data[['trade_date', 'close']]
    data.columns = ['ds', 'y']

    # 初始化Prophet模型并训练
    model = Prophet(daily_seasonality=True)
    model.fit(data)

    # 构建未来数据框架并进行预测
    future = model.make_future_dataframe(periods=steps_to_predict)
    forecast = model.predict(future)

    # 提取预测结果
    predictions = forecast[['ds', 'yhat']].tail(steps_to_predict)

    return predictions

if __name__ == "__main__":
    n_step = 7
    count = 200
    data = pd.read_csv('../data/000001.SZ.csv').iloc[:count, :]
    # 将data反转，使其按照时间顺序排列
    data = data.iloc[::-1]

    train_data = data.iloc[:-n_step, :]
    test_data = data.iloc[-n_step:, :]
    pred = forecast(train_data, n_step)
    print(count)
    evaluate(test_data['close'], pred['yhat'])
    showTruePred(test_data['close'], pred['yhat'])


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