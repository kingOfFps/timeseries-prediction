from django.urls import path
from django.urls import re_path as url

from . import views

urlpatterns = [
    # url(r'^stock/$', views.stockList, name='stockList'),
    # url(r'^stock/(?P<stock_id>\d+)/$', views.stockDetail, name='stockDetail'),

    path('stock/', views.stock_list, name='stock_list'),
    path('stock/filter_data/', views.filter_data, name='filter_data'),
    path('stock/<str:ts_code>/', views.stock_detail, name='stock_detail'),

    # path("chart/list/", views.chart_list),
    path("stock/chart/line/", views.chart_line),
    # path('filter_stock_data/', views.filter_stock_data, name='filter_stock_data'),
]
#
# import baostock as bs
# import pandas as pd
#
# code = 'sh.600036'
# start_date = '2000-01-01'
# end_date = '2021-10-01'
#
# # Step1： 获取数据
# lg = bs.login()
# rs = bs.query_history_k_data_plus(code,
#                                   "date,code,open,high,low,close,volume",
#                                   start_date=start_date, end_date=end_date, frequency="d",
#                                   adjustflag='2')  # 注意adjustflag取前复权
# data_list = []
# while (rs.error_code == '0') & rs.next():
#     data_list.append(rs.get_row_data())
# df = pd.DataFrame(data_list, columns=rs.fields)
# df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(
#     'float64')
# df = df.rename(columns={'date': 'datetime'})
# df.index = pd.DatetimeIndex(df['datetime'])
# bs.logout()
#
# import time
#
# time_start = time.time()
#
# # Step2： 利用Pandas 计算MACD
#
# short_ema = df['close'].ewm(span=12).mean()
# long_ema = df['close'].ewm(span=26).mean()
# df.loc[:, 'DIFF'] = short_ema - long_ema
# df.loc[:, 'DEA'] = df['DIFF'].ewm(span=9).mean()
# df.loc[:, 'MACD'] = 2 * (df['DIFF'] - df['DEA'])
#
#
# time_end = time.time()
# print('pandas totally cost:', time_end - time_start)
