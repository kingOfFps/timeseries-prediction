import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
import itertools
import warnings
import joblib
from forecast.utils import *


def confirm_p_q(data):
    warnings.filterwarnings('ignore')
    """为了快速训练，max_ar和max_ma取的是1，正常情况应该都取4"""
    AIC = sm.tsa.arma_order_select_ic(data, max_ar=4, max_ma=4, ic='aic')['aic_min_order']
    print(f'AIC:{AIC}')
    print('AIC：', AIC)
    return AIC



def forecast(data,n_step):
    # data = pd.Series(np.array(data[:,-1]), index=data['datetime'])
    order = confirm_p_q(data)
    model = sm.tsa.ARIMA(data, order=(order[0], 1, order[1]))
    model = model.fit()
    pred = model.forecast(n_step)
    return pred

if __name__ == "__main__":
    n_step = 7
    data = pd.read_csv('../data/000001.SZ.csv')
    train_data = data.iloc[:-n_step, :]
    test_data = data.iloc[-n_step:, :]
    pred = forecast(train_data['close'], n_step)

    evaluate(test_data['close'], pred)
    showTruePred(test_data['close'], pred)