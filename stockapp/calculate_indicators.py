
import numpy as np
import pandas as pd


def MACD(data, fast=12, slow=26, signal=9):
    """
       计算MACD指标。
       :param data: 包含价格数据的dataframe，必须包含'close'列。
       :param fast: 快速移动平均线的时间窗口。
       :param slow: 慢速移动平均线的时间窗口。
       :param signal: 信号线的时间窗口。
       :return: 包含MACD指标和信号线的dataframe。
       在MACD指标中，信号线(signal line)是MACD指标的移动平均线，用于衡量趋势的变化。
       直方图(histogram)是MACD指标和信号线之间的差异，用于衡量市场的动能(momentum)。
       """
    close = data['close']
    ema_fast = close.ewm(span=fast, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, min_periods=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, min_periods=signal).mean()
    histogram = macd - signal_line
    data['MACD'] = macd
    data['Signal_Line'] = signal_line
    data['Histogram'] = histogram
    data['MACD'].fillna(method='bfill', inplace=True)
    data['Signal_Line'].fillna(method='bfill', inplace=True)
    data['Histogram'].fillna(method='bfill', inplace=True)
    # print("###")
    # return macd.to_list(), signal_line.to_list(), histogram.to_list()
    return data


def KDJ(data, n=9, m1=3, m2=3):
    """
       计算KDJ指标。
       :param data: 包含价格数据的dataframe，必须包含'High'、'Low'、'Close'列。
       :param n: 计算KDJ指标所需的时间窗口长度。
       :param m1: 计算KDJ指标的平滑因子1的时间窗口长度。
       :param m2: 计算KDJ指标的平滑因子2的时间窗口长度。
       :return: 包含KDJ指标的dataframe。
       """
    high = data['high']
    low = data['low']
    close = data['close']
    rsv = (close - low.rolling(n).min()) / (high.rolling(n).max() - low.rolling(n).min()) * 100
    k = rsv.ewm(alpha=1 / m1, min_periods=m1).mean()
    d = k.ewm(alpha=1 / m2, min_periods=m2).mean()
    j = 3 * k - 2 * d
    data['K'] = k
    data['K'].fillna(method='bfill', inplace=True)
    data['D'] = d
    data['D'].fillna(method='bfill', inplace=True)
    data['J'] = j
    data['J'].fillna(method='bfill', inplace=True)
    return data


def W_And_M(data, n=14):
    """
       计算威廉指标（W&M）。
       :param data: 包含价格数据的dataframe，必须包含'High'、'Low'、'Close'列。
       :param n: 计算威廉指标所需的时间窗口长度。
       :return: 包含威廉指标的dataframe。
       """
    high = data['high']
    low = data['low']
    close = data['close']
    wr = (high.rolling(n).max() - close) / (high.rolling(n).max() - low.rolling(n).min()) * -100
    data['Williams_%R'] = wr
    data['Williams_%R'].fillna(method='bfill', inplace=True)
    return data


def RSI(data, n=14):
    """
    计算RSI指标。
    :param data: 包含价格数据的dataframe，必须包含'Close'列。
    :param n: 计算RSI指标所需的时间窗口长度。
    :return: RSI值的列表
    """
    close = data['close']
    delta = close.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up = up.rolling(n).mean()
    roll_down = abs(down.rolling(n).mean())
    rs = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rs))
    data['RSI'] = rsi
    data['RSI'].fillna(method='bfill', inplace=True)
    return data


def DMI(data, n=14, m=14):
    """
    计算DMI指标。
    :param data: 包含价格数据的dataframe，必须包含'High'、'Low'、'Close'列。
    :param n: 计算DMI指标所需的时间窗口长度。
    :param m: 计算DMI指标的平滑因子的时间窗口长度。
    :return: 包含DMI指标和ADX指标的dataframe。
    """
    high = data['high']
    low = data['low']
    close = data['close']
    tr = pd.DataFrame(np.zeros_like(close), index=data.index, columns=['TR'])
    tr['H-L'] = abs(high - low)
    tr['H-PC'] = abs(high - close.shift(1))
    tr['L-PC'] = abs(low - close.shift(1))
    tr['TR'] = tr[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    atr = tr['TR'].rolling(n).mean()
    up = high.diff()
    down = low.diff()
    plus_dm = pd.DataFrame(np.zeros_like(close), index=data.index, columns=['+DM'])
    plus_dm[(up > 0) & (up > down)] = up
    minus_dm = pd.DataFrame(np.zeros_like(close), index=data.index, columns=['-DM'])
    minus_dm[(down > 0) & (down > up)] = down
    plus_di = 100 * plus_dm['+DM'].rolling(n).sum() / atr
    minus_di = 100 * minus_dm['-DM'].rolling(n).sum() / atr
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/m, min_periods=m).mean()
    data['DI+'] = plus_di
    data['DI+'].fillna(method='bfill', inplace=True)
    data['DI-'] = minus_di
    data['DI-'].fillna(method='bfill', inplace=True)
    data['ADX'] = adx
    data['ADX'].fillna(method='bfill', inplace=True)

    return data


def BIAS(data, n=6):
    """
    计算BIAS指标。
    :param data: 包含价格数据的dataframe，必须包含'Close'列。
    :param n: 计算BIAS指标所需的时间窗口长度。
    :return: 包含BIAS指标的dataframe。
    """
    close = data['close']
    ma = close.rolling(n).mean()
    bias = (close - ma) / ma * 100
    data['BIAS'] = bias
    data['BIAS'].fillna(method='bfill', inplace=True)
    return data


def OBV(data):
    """
    计算OBV指标。
    :param data: 包含价格数据和成交量数据的dataframe，必须包含'Close'和'Volume'列。
    :return: 包含OBV指标的dataframe。
    """
    close = data['close']
    volume = data['vol']
    obv = pd.Series(np.zeros_like(close), index=data.index)
    obv[0] = np.sign(close[1] - close[0]) * volume[0]
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - volume[i]
        else:
            obv[i] = obv[i-1]
    data['OBV'] = obv
    data['OBV'].fillna(method='bfill', inplace=True)

    return data


def BOLL(data, n=20, k=2):
    """
    计算BOLL指标。
    :param data: 包含价格数据的dataframe，必须包含'Close'列。
    :param n: 计算移动平均线所需的时间窗口长度。
    :param k: 计算标准差倍数。
    :return: 包含BOLL指标的dataframe。
    """
    close = data['close']
    ma = close.rolling(n).mean()
    std = close.rolling(n).std()
    up = ma + k * std
    down = ma - k * std
    data['BOLL'] = ma
    data['BOLL'].fillna(method='bfill', inplace=True)
    data['BOLL_UPPER'] = up
    data['BOLL_UPPER'].fillna(method='bfill', inplace=True)
    data['BOLL_LOWER'] = down
    data['BOLL_LOWER'].fillna(method='bfill', inplace=True)

    return data


def calculate_main(data):
    data = MACD(data)
    data = KDJ(data)
    data = W_And_M(data)
    data = RSI(data)
    # data = DMI(data)
    data = BIAS(data)
    data = OBV(data)
    data = BOLL(data)

    return data
