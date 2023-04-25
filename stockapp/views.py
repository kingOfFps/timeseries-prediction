from django.core.paginator import Paginator
import datetime
import locale
import json
import joblib
from django.shortcuts import render
from . import *
from django.http import JsonResponse
import forecast.linearRegression as linearRegression
import forecast.lstm as lstm
from stockapp.calculate_indicators import *


def stock_detail(request, ts_code):
    """根据请求的ts_code，显示股票的详情页"""
    row_data = df[df['ts_code'] == ts_code]
    row_data.index.name = 'index'
    return render(request, 'stockapp/stockDetail.html', {'data': row_data.to_dict(orient='records')[0]})


def updateAllStock(request):
    """更新所有股票信息到data/xxxx.csv,也就是更新单个股票的详细信息中(异步请求接口)"""
    # 接口使用方法：https://tushare.pro/document/2?doc_id=25
    # 声明为全局变量，使得后续对df的操作能覆盖全局变量   5158
    global df
    # 先更新所有股票的简要信息（allStock.csv），在根据allStock.csv中的ts_code挨个去更行 ts_code.csv，也就是股票的详细信息
    df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
    # 给股票的简要信息添加如下几列
    for col_name in add_list:
        df[col_name] = None

    # for ts_code in df['ts_code'].to_list():
    for ts_code in df.loc[:, 'ts_code'].to_list():
        data = pro.daily(ts_code=ts_code)
        data.to_csv(f'data/{ts_code}.csv')
        # 将ts_code.csv的详情数据添加到allStock.csv中
        if data.shape[0] < 1:
            continue
        for col_name in add_list:
            if col_name in data.columns:
                df.loc[df['ts_code'] == ts_code, col_name] = data.loc[0, col_name]
    df.to_csv('data/allStock.csv')
    return JsonResponse({'status': True, 'msg': '所有股票数据，更新成功'})


def chart_line(request):
    ts_code = request.GET.get('ts_code')
    start_date = request.GET.get('start_date', '20190120')  # Use a default value if not provided
    end_date = request.GET.get('end_date', None)  # Use None if not provided

    # Pass start_date and end_date to pro.daily()
    data = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    # 将 DataFrame 逆序,这样折线图的x（时间才是从小到大）
    data = data.iloc[::-1]

    data = calculate_main(data)
    legend = ["open", "high", "low", "close", "change", "vol", "amount", 'MACD', 'MACD-signal-line', 'MACD-histogram',
              'KDJ-K', 'KDJ-D', 'KDJ-J', 'W&M', 'RSI', 'DI+', 'DI-', 'ADX', 'BIAS', 'BOLL-MA', 'BOLL-UP', 'BOLL-DOWN']
    series_list = [
        # 0 开盘价
        {"name": 'open', "type": 'line', "stack": 'Total', 'smooth': True, "data": data['open'].to_list(), },
        # 1 当天最高
        {"name": 'high', "type": 'line', "stack": 'Total', 'smooth': True, "data": data['high'].to_list(), },
        # 2 当天最低
        {"name": 'low', "type": 'line', "stack": 'Total', 'smooth': True, "data": data['low'].to_list(), },
        # 3 收盘价
        {"name": 'close', "type": 'line', "stack": 'Total', 'smooth': True, "data": data['close'].to_list(), },
        # 4 当日变化
        {"name": 'change', "type": 'line', "stack": 'Total', 'smooth': True, "data": data['change'].to_list(), },
        # 5 成交量
        {"name": 'vol', "type": 'line', "stack": 'Total', 'smooth': True, "data": data['vol'].to_list(), },
        # 6 成交额
        {"name": 'amount', "type": 'line', "stack": 'Total', 'smooth': True, "data": data['amount'].to_list(), },
        # 7 MACD线
        {"name": 'MACD', "type": 'line', "stack": 'Total', 'smooth': True, "data": data['MACD'].to_list(), },
        # 8 MACD信号线  也是移动平均线 窗口大小为9
        {"name": 'MACD-signal-line', "type": 'line', "stack": 'Total', 'smooth': True,
         "data": data['Signal_Line'].to_list(), },
        # 9 MACD直方图  MACD与信号线之间的差距
        {"name": 'MACD-histogram', "type": 'line', "stack": 'Total', 'smooth': True,
         "data": data['Histogram'].to_list(), },
        # 10 KDJ的K线
        {"name": 'KDJ-K', "type": 'line', "stack": 'Total', 'smooth': True, "data": data['K'].to_list(), },
        # 11 KDJ的D线
        {"name": 'KDJ-D', "type": 'line', "stack": 'Total', 'smooth': True, "data": data['D'].to_list(), },
        # 12 KDJ的J线
        {"name": 'KDJ-J', "type": 'line', "stack": 'Total', 'smooth': True, "data": data['J'].to_list(), },
        # 13 威廉指标 W&M
        {"name": 'W&M', "type": 'line', "stack": 'Total', 'smooth': True, "data": data['Williams_%R'].to_list(), },
        # 14 RSI
        {"name": 'RSI', "type": 'line', "stack": 'Total', 'smooth': True, "data": data['RSI'].to_list(), },
        # 15 DMI-正向动向指标
        # {"name": 'DI+', "type": 'line', "stack": 'Total', 'smooth': True, "data": data['RSI'].to_list(), },
        # 16 DMI-负向动向指标
        # {"name": 'DI-', "type": 'line', "stack": 'Total', 'smooth': True, "data": data['RSI'].to_list(), },
        # 17 DMI-动向指数
        # {"name": 'ADX', "type": 'line', "stack": 'Total', 'smooth': True, "data": data['RSI'].to_list(), },
        # 15 BIAS
        {"name": 'BIAS', "type": 'line', "stack": 'Total', 'smooth': True, "data": data['BIAS'].to_list(), },
        # 16 BOLL指标
        {"name": 'BOLL-MA', "type": 'line', "stack": 'Total', 'smooth': True, "data": data['BOLL'].to_list(), },
        # 17 BOLL上轨
        {"name": 'BOLL-UP', "type": 'line', "stack": 'Total', 'smooth': True, "data": data['BOLL_UPPER'].to_list(), },
        # 18 BOLL下轨
        {"name": 'BOLL-DOWN', "type": 'line', "stack": 'Total', 'smooth': True, "data": data['BOLL_LOWER'].to_list(), },
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
    """
    需求：Django项目中，我现在有所有股票的简要信息allStock.csv,和allStock.csv的 ts_code列出现的股票详细信息 ts_code.csv。我现在需要通过一个
    view函数，将

    """
    """获取股票数据，返回给stockList.html显示"""
    filter_options = {
        'area': df['area'].unique().tolist(),
        'industry': df['industry'].unique().tolist(),
        # 'market': df['market'].unique().tolist(),
    }
    data = df.to_dict(orient='records')
    # 创建分页器
    paginator = Paginator(data, page_num)
    # 从请求中获取页码
    page = request.GET.get('page', 1)
    # 获取当前页的数据
    page_data = paginator.get_page(page)
    return render(request, 'stockapp/stockList.html', {'data': page_data, 'filter_options': filter_options})


def filter_options(request):
    """查询股票数据中对应的地区、行业、市场，返回给stockList.html,让其下拉框显示这些数据"""
    # 设置当前区域设置为中国
    locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')
    areas = list(df['area'].dropna().unique())
    industries = list(df['industry'].dropna().unique())
    # markets = list(df['market'].dropna().unique())

    return JsonResponse({
        'area': areas,
        'industry': industries,
        # 'market': markets
    })


def filter_data(request):
    """根据 stockList.html给出的筛选条件，筛选出对应的股票数据返回"""
    if request.method == 'POST':
        filters = request.POST.get('filters')
        filters = json.loads(filters)
        filtered_data = df.copy()
        filtered_data.dropna(inplace=True)
        for column, value in filters.items():
            if value:
                if column == 'list_date':
                    start_date, end_date = value.split(' - ')
                    start_date = int(start_date.replace('-', ''))
                    end_date = int(end_date.replace('-', ''))
                    filtered_data = filtered_data[
                        (filtered_data[column] >= start_date) & (filtered_data[column] <= end_date)]
                else:
                    filtered_data = filtered_data[filtered_data[column].str.contains(value)]
        data = filtered_data.to_dict(orient='records')
        # 创建分页器
        paginator = Paginator(data, page_num)
        # 从请求中获取页码
        page = request.GET.get('page', 1)
        # 获取当前页的数据
        page_data = paginator.get_page(page)

        # 提取分页数据和分页信息
        response_data = {
            'data': list(page_data.object_list),
            'number': page_data.number,
            'has_previous': page_data.has_previous(),
            'has_next': page_data.has_next(),
            'previous_page_number': page_data.previous_page_number() if page_data.has_previous() else None,
            'next_page_number': page_data.next_page_number() if page_data.has_next() else None,
            'start_index': page_data.start_index(),
            'end_index': page_data.end_index(),
        }

        return JsonResponse(response_data)
        # return JsonResponse({'data': data})
    else:
        return JsonResponse({'error': 'Invalid request method'})


def trainLR(request):
    """针对allStock.csv的前面100支股票的历史数据，训练其对应的线性回归模型"""

    for ts_code in df.loc[:stock_count, 'ts_code'].to_list():
        pass


def predict(request):
    ts_code = request.GET.get('ts_code')
    # 预测的天数
    n_step = request.GET.get('n_step')
    data = pro.daily(ts_code=ts_code)
    X = data[['close']][-n_step:]
    path = f'forecast/model/linearRegression.pkl'
    model = joblib.load(path)
    result = linearRegression.predict(model, X=X, window_size=7)


def predictOnline(request):
    """在线预测allStock.csv中的前{stock_count}支股票的{n_step}后的收盘价，
    将每只股票的预测的收盘价-现在的开盘价存入result_list中，就能得到这些股票的收益了"""
    from . import count
    result_list = []
    for ts_code in df.loc[:stock_count, 'ts_code'].to_list():
        data = pd.read_csv('../data/000001.SZ.csv')
        count = min(count, data.shape[0])
        data = pd.read_csv('../data/000001.SZ.csv').iloc[:count, :]

    # 预测的天数
    n_step = request.GET.get('n_step')
    data = pro.daily(ts_code=ts_code)
    X = data[['close']][-n_step:]
    path = f'forecast/model/linearRegression.pkl'
    model = joblib.load(path)
    result = linearRegression.predict(model, X=X, window_size=7)


"""
需求阐述：
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
