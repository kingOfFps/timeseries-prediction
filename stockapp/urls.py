from django.urls import path
from django.urls import re_path as url

from . import views

urlpatterns = [
    # url(r'^stock/$', views.stockList, name='stockList'),
    # url(r'^stock/(?P<stock_id>\d+)/$', views.stockDetail, name='stockDetail'),
    # path('filter_stock_data/', views.filter_stock_data, name='filter_stock_data'),

    path('stock/', views.stock_list, name='stock_list'),
    path('stock/filter_data/', views.filter_data, name='filter_data'),
    path('stock/filter_options/', views.filter_options, name='filter_options'),


    path("stock/chart/line/", views.chart_line),
    path("stock/chart/marketline/", views.market_line),

    path("stock/actions/updateAllStock/", views.updateAllStock),
    path("stock/actions/predict/stocks", views.predictStockList),
    path("stock/actions/predict/<str:ts_code>", views.predictSingleStock),
    path("stock/actions/getStockData/<str:ts_code>", views.getStockData),

    path('stock/stock_comparison/', views.stock_comparison, name='stock_comparison'),
    path('stock/market_index/', views.market_index, name='market_index'),
    path('stock/<str:ts_code>/', views.stock_detail, name='stock_detail'),


]


"""


基于keras用LSTM实现股票收盘价的预测，数据格式如下：“,ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount
0,600000.SH,20230418,7.38,7.59,7.37,7.54,7.39,0.15,2.0298,774123.35,582499.848”
预测的时候用到特征：“open,high,low,close,pre_close,change,pct_chg,vol,amount”。

"""


