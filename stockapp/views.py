import datetime
import locale

import json
import pandas as pd
from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
# import tushare as ts
from . import pro, df
import os

# stockapp/views.py
from django.http import JsonResponse


def get_filtered_stock_data(stock_code, start_date, end_date, open_price_range, close_price_range, volume_range):
    # df = ts.get_hist_data(stock_code, start=start_date, end=end_date)
    df = pro.daily(ts_code=stock_code, start_date='20190120')

    if open_price_range:
        df = df[(df['open'] >= open_price_range[0]) & (df['open'] <= open_price_range[1])]
    if close_price_range:
        df = df[(df['close'] >= close_price_range[0]) & (df['close'] <= close_price_range[1])]
    if volume_range:
        df = df[(df['volume'] >= volume_range[0]) & (df['volume'] <= volume_range[1])]
    return df


def filter_stock_data(request):
    stock_code = request.GET.get('stock_code', '600000.SH')
    start_date = request.GET.get('start_date', '2021-01-01')
    end_date = request.GET.get('end_date', '2021-12-31')
    open_price_range = request.GET.get('open_price_range', None)
    close_price_range = request.GET.get('close_price_range', None)
    volume_range = request.GET.get('volume_range', None)

    if open_price_range:
        open_price_range = list(map(float, open_price_range.split(',')))
    if close_price_range:
        close_price_range = list(map(float, close_price_range.split(',')))
    if volume_range:
        volume_range = list(map(float, volume_range.split(',')))

    stock_data = get_filtered_stock_data(stock_code, start_date, end_date, open_price_range, close_price_range,
                                         volume_range)
    stock_data.reset_index(inplace=True)
    return JsonResponse(stock_data.to_dict(orient='list'), safe=False)


def stock_data(request):
    code = '600000.SH'
    # 获取股票数据
    # df = pro.daily_basic(ts_code=code, start_date='20190120')  这个接口积分不够，无法调用

    df = pro.daily(ts_code=code, start_date='20190120')

    df.to_csv(f'data/{code}.csv',index=False)
    # 将数据传递给模板
    context = {'data': df.to_dict()}
    return render(request, 'stockapp/stock_data.html', context)


def get_stock_data(stock_code, start_date, end_date):
    df = pro.daily(ts_code=stock_code, start_date=start_date)
    df.to_csv(f'data/{stock_code}.csv')
    return df


# def stockList(request):
#     """当get中有筛选条件时，展示指定条件的股票列表；没有条件时，则展示所有列表"""
#
#     """获取所有股票的Dataframe对象"""
#     df = pd.read_csv('data/allStock.csv')
#     """按照条件进行筛选，无条件则默认给个条件或者展示所有股票列表。可供筛选的参数：
#     ts_code    name     area industry    list_date"""
#     # 将 request.GET 转换为字典
#     params = dict(request.GET.items())
#     for k, v in params.items():
#         if k in {'ts_code','name', 'area', 'industry'}:
#             df = df[df[k] == v]
#
#     if 'start_date' in params.keys():
#         df = df[(df['list_date'] >= int(params['start_date']))]
#     if 'end_date' in params.keys():
#         df = df[(df['list_date'] <= int(params['end_date']))]
#     columns = df.columns.tolist()
#     columns[0] = '序号'
#     count = min(df.shape[0],100)
#     data = df.iloc[:count,:].values.tolist()
#
#     return render(request, 'stockapp/stockList.html', {'columns': columns, 'data': data})
#     # return render(request, 'stockapp/stockList.html', {'res': result})
#     #
#     # result = {
#     #     "status": True,
#     #     "data": {
#     #         'stockList': list(df),
#     #     },
#     # }
#     # stock_code = '600000.SH'  # Example stock code
#     # start_date = '2021-01-01'
#     # end_date = '2021-12-31'
#     # stock_data = get_stock_data(stock_code, start_date, end_date)
#     # return render(request, 'stockapp/index.html', {'stock_data': stock_data})


def stock_detail(request, ts_code):
    row_data = df[df['ts_code'] == ts_code]
    row_data.index.name = 'index'
    return render(request, 'stockapp/stockDetail.html', {'data': row_data.to_dict(orient='records')[0]})


def updateAllStock(request):
    """更新所有股票信息到data/allStock.csv中"""
    # 接口使用方法：https://tushare.pro/document/2?doc_id=25
    try:
        data = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
        data.to_csv('data/allStock.csv')
        return JsonResponse({'status': True, 'msg': '更新成功'})
    except:
        return JsonResponse({'status': False, 'msg': '更新失败'})


def chart_line(request):
    ts_code = request.GET.get('ts_code')
    start_date = request.GET.get('start_date', '20190120')  # Use a default value if not provided
    end_date = request.GET.get('end_date', None)  # Use None if not provided

    # Pass start_date and end_date to pro.daily()
    data = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    # 将 DataFrame 逆序,这样折线图的x（时间才是从小到大）
    data = data.iloc[::-1]

    vol = data['vol'].to_list()
    amount = data['amount'].to_list()
    legend = ['volume', 'amount']
    series_list = [
        {
            "name": 'volume',
            "type": 'line',
            "stack": 'Total',
            'smooth': True,
            "data": vol,
        },
        {
            "name": 'amount',
            "type": 'line',
            "stack": 'Total',
            "data": amount,
        }
    ]
    x_axis = data['trade_date'].to_list()

    result = {
        "status": True,
        "data": {
            'legend': legend,
            'series_list': series_list,
            'x_axis': x_axis,
        },
    }
    return JsonResponse(result)


def stock_list(request):
    filter_options = {
        'area': df['area'].unique().tolist(),
        'industry': df['industry'].unique().tolist(),
        'market': df['market'].unique().tolist(),
    }
    count = min(df.shape[0], 100)
    # data = df.to_dict(orient='records')[:count]
    data = df.to_dict(orient='records')
    return render(request, 'stockapp/stockList.html', {'data': data, 'filter_options': filter_options})

def filter_options(request):
    # 设置当前区域设置为中国
    locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')
    # areas = sorted(list(df['area'].unique()), key=locale.strxfrm)
    # industries = sorted(list(df['industry'].unique()), key=locale.strxfrm)
    # markets = sorted(list(df['market'].unique()), key=locale.strxfrm)

    areas = list(df['area'].dropna().unique())
    industries = list(df['industry'].dropna().unique())
    markets = list(df['market'].dropna().unique())

    return JsonResponse({
        'area': areas,
        'industry': industries,
        'market': markets
    })

# def filter_data(request):
#     if request.method == 'POST':
#         filters = request.POST.get('filters')
#         filters = json.loads(filters)
#
#         filtered_data = df.copy()
#         for column, value in filters.items():
#             if value != "":
#                 if column == 'list_date':
#                     start_date, end_date = value.split(' - ')
#                     filtered_data = filtered_data[
#                         (filtered_data[column] >= start_date) & (filtered_data[column] <= end_date)]
#                 else:
#                     filtered_data = filtered_data[filtered_data[column] == value]
#
#         return JsonResponse({'data': filtered_data.to_dict(orient='records')})
#     else:
#         return JsonResponse({'error': 'Invalid request method'})


def filter_data(request):
    if request.method == 'POST':
        filters = request.POST.get('filters')
        filters = json.loads(filters)

        filtered_data = df.copy()
        filtered_data.dropna(inplace=True)
        for column, value in filters.items():
            if value:
                if column == 'list_date':
                    start_date, end_date = value.split(' - ')
                    start_date = int(start_date.replace('-',''))
                    end_date = int(end_date.replace('-',''))
                    filtered_data = filtered_data[
                        (filtered_data[column] >= start_date) & (filtered_data[column] <= end_date)]
                else:
                    filtered_data = filtered_data[filtered_data[column].str.contains(value)]

        return JsonResponse({'data': filtered_data.to_dict(orient='records')})
    else:
        return JsonResponse({'error': 'Invalid request method'})


"""
在django框架中，我现在有dataframe数据，部分数据展示：“,ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount
0,600000.SH,20230418,7.38,7.59,7.37,7.54,7.39,0.15,2.0298,774123.35,582499.848”，
我需要一个view函数来将dataframe传递给stockList.html来展示数据。需要另一个view函数来处理stockList.html对展示的dataframe数据进行的筛选操作，
按照前端发来的异步请求筛选对应的dataframe数据，并返回给stockList.html展示。
要求：请帮我基于Bootstrap4和Django3开发出美观的html页面，用表格在stockList.html来展示这些dataframe数据，并在页面中为dataframe的每一列加上筛选操作。
并且将每行的‘ts_code’列写成超链接的形式，链接形式：“/stock/{ts_code}”，来跳转到stockDetail.html页面。stockDetail.html根据超链接中的{ts_code}展示某行dataframe的详细信息。
请帮我编写如下代码：3个view函数，分别用于在stockList.html展示dataframe、筛选dataframe、跳转到dataframe行对应的stockDetail.html。
stockList.html,stockDetail.html。js直接写入html文件中，不要单独写成一个js文件。html一定要美观好看。


我希望：1 stockList.html的数据列表中，name、area、industry、market的筛选方式是下拉框而不是输入框。并且所有下拉框的值是通过一个异步接口获取到的，
步骤就是在view函数中对对应dataframe列的值进行去重操作，去重后的值就是对应列的下拉框值。
2. list_date的筛选方式通过前端的日期组件来选择开始日期和和结束日期而不是输入框。
3. stockList.html的数据列表太宽，stockDetail也不够漂亮，基于bootstrap4帮我美化他们

"""
