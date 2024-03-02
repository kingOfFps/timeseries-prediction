import pandas as pd
"""
对原始的数据集进行各种处理，包括缺失值填补、异常检测、重采样等。处理一次后就可以被各个算法来用做实验了。
这个过程是在各个算法实验之前的。
"""


"""
我现在有一个csv的时间序列数据集，示例如下：”datetime	temperature	humidity	wind_speed	rainfall	load
2020/1/1 0:00	14.29	47.97	3.5	0	45208.293
2020/1/1 0:15	14.12	48.42	3.3	0	44342.25
2020/1/1 0:30	13.96	48.85	3.07	0	43726.9609
2020/1/1 0:45	13.83	49.15	2.91	0	43055.1133
2020/1/1 1:00	13.73	49.38	2.84	0	42335.375
2020/1/1 1:15	13.62	49.56	2.89	0	41823.9766
“，数据集的时间粒度是15mins，我希望对数据集进行重采样，将其粒度变为1小时。对所有维度的重采样的计算方式为平均值重采样。
请帮我按照python函数的形式给出代码

"""

def resample_data(filename):
    # 读取CSV文件，并将'datetime'列解析为日期时间类型
    # df = pd.read_csv(filename, parse_dates=['datetime'])
    df = pd.read_csv(filename,
                     parse_dates=['datetime'], infer_datetime_format=True,
                     low_memory=False, na_values=['nan', '?'], index_col='datetime')
    # 进行1小时的平均值重采样
    df_resampled = df.resample('1H').mean()
    df_resampled.to_csv('../../data/agriculture_load_h.csv')
    return df_resampled

if __name__ == '__main__':
    resample_data('../../data/agriculture_load.csv')
