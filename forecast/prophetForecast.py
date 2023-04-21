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
    n_step = 50
    data = pd.read_csv('../data/600000.SH.csv')
    train_data = data.iloc[:-n_step, :]
    test_data = data.iloc[-n_step:, :]
    pred = forecast(train_data, n_step)

    evaluate(test_data['close'], pred['yhat'])
    showTruePred(test_data['close'], pred['yhat'])