import time
from django.contrib.auth.decorators import login_required

from django.core.paginator import Paginator
import datetime
import locale
import json
import joblib
from django.shortcuts import render
from . import *
from django.http import JsonResponse
from datetime import datetime, timedelta

import forecast.lstm_single as lstm
import forecast.lstm_mutiple as lstm_mutiple
from stockapp.calculate_indicators import *


# @login_required(login_url='/login/')
def stock_detail(request, ts_code):
    row_data = getData(ts_code)
    return render(request, 'stockapp/stockDetail.html', {'data': row_data})


def getData(ts_code):
    """根据请求的ts_code，返回股票的详情页数据"""
    row_data = df[df['ts_code'] == ts_code]
    row_data.index.name = 'index'
    row_data = row_data.to_dict(orient='records')[0]
    if 'Unnamed: 0' in row_data.keys():
        row_data.pop('Unnamed: 0')
    return row_data


def getStockData(request, ts_code):
    """根据请求的ts_code，返回股票的详情页数据"""
    data = getData(ts_code)
    return JsonResponse({'data': data})


def updateAllStock(request):
    """更新所有股票信息到data/xxxx.csv,也就是更新单个股票的详细信息中(异步请求接口)"""
    # 接口使用方法：https://tushare.pro/document/2?doc_id=25
    # 声明为全局变量，使得后续对df的操作能覆盖全局变量   5158
    global df
    # 先更新所有股票的简要信息（allStock.csv），在根据allStock.csv中的ts_code挨个去更行 ts_code.csv，也就是股票的详细信息
    df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date').iloc[
         :config['stock_count'], ]
    # 给股票的简要信息添加如下几列
    for col_name in config['add_list']:
        df[col_name] = None

    # for ts_code in df['ts_code'].to_list():
    for ts_code in df.loc[:, 'ts_code'].to_list():
        data = pro.daily(ts_code=ts_code)
        data.to_csv(f'data/{ts_code}.csv', index=False)
        # 将ts_code.csv的详情数据添加到allStock.csv中
        if data.shape[0] < 1:
            continue
        for col_name in config['add_list']:
            if col_name in data.columns:
                df.loc[df['ts_code'] == ts_code, col_name] = data.loc[0, col_name]
    df.to_csv('data/allStock.csv', index=False)
    return JsonResponse({'status': True, 'msg': '所有股票数据，更新成功'})


def chart_line(request):
    ts_code = request.GET.get('ts_code')
    start_date = request.GET.get('start_date', '20220101')  # Use a default value if not provided
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


# @login_required(login_url='/login/')
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
    paginator = Paginator(data, config['page_num'])
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
        paginator = Paginator(data, config['page_num'])
        # 从请求中获取页码
        page = request.GET.get('page', 1)
        # 获取当前页的数据
        page_data = paginator.get_page(page)

        # 提取分页数据和分页信息
        response_data = {
            'data': list(page_data.object_list),
            'number': page_data.number,
            'has_previous': page_data.has_previous(),
            'has_next': page_data.hFas_next(),
            'previous_page_number': page_data.previous_page_number() if page_data.has_previous() else None,
            'next_page_number': page_data.next_page_number() if page_data.has_next() else None,
            'start_index': page_data.start_index(),
            'end_index': page_data.end_index(),
        }

        return JsonResponse(response_data)
        # return JsonResponse({'data': data})
    else:
        return JsonResponse({'error': 'Invalid request method'})


def predictStockList(request):
    """在线预测allStock.csv中的前{predict_stock_count}支股票的{n_step}后的收盘价，
    将每只股票的预测的收盘价-现在的开盘价存入result_list中，就能得到这些股票的收益了"""
    start_time = time.time()
    # 预测的天数
    n_step = request.GET.get('n_step', 5)
    if n_step == '' or n_step is None:
        n_step = 3
    n_step = int(n_step)

    results = []
    for ts_code in df.loc[:config['predict_stock_count'], 'ts_code'].to_list():
        dic = {'ts_code': None, 'name': None, 'open': None, 'close': None, 'change': None, 'earn': None}
        data = pd.read_csv(f'../data/{ts_code}.csv')
        count = min(config['predict_count'], data.shape[0])
        # 获取部分数据并反转顺序
        data = data.iloc[:count, :].iloc[::-1]
        pred = lstm.forecast(data, n_step)
        # 将股票数据序列化到字典中
        close = float(pred[-1])
        open = data['open'].iloc[-1]
        earn = close - open
        change = '%.4f' % (earn / open * 100)

        dic['ts_code'] = ts_code
        dic['name'] = df.loc[df['ts_code'] == ts_code, 'name'].iloc[0]
        dic['open'] = open
        dic['close'] = "%.4f" % close
        dic['earn'] = "%.4f" % earn
        dic['change'] = f"{change}%"
        results.append(dic)
    results = sorted(results, key=lambda x: float(x['earn']), reverse=True)
    print(f'预测花费{time.time() - start_time}')
    return render(request, 'stockapp/predictList.html', {'data': results})
    # return JsonResponse({'data': results})


def predictSingleStock(request, ts_code):
    """针对单只股票的最高价、最低价、收盘价、长跌幅进行预测，预测步数是未来七天"""
    data = pd.read_csv(f'./data/{ts_code}.csv')
    count = min(config['predict_count'], data.shape[0])
    # 获取部分数据并反转顺序
    data = data.iloc[:count, :].iloc[::-1]
    # 调用lstm对股票未来7天的数据进行预测。预测返回的是一个形状为(7,4)的ndarray对象。
    predictions = lstm_mutiple.predict(data, config['step_in'], config['n_step'], f'forecast/model/{ts_code}')
    historical_data = data.iloc[-config['n_step'] * 2:]

    # 将 trade_date 列转换为日期，并将其设置为索引
    historical_data['trade_date'] = pd.to_datetime(historical_data['trade_date'], format='%Y%m%d')
    historical_data.set_index('trade_date', inplace=True)

    # 创建一个新的 DataFrame，用于存储预测数据
    pred_columns = ['high', 'low', 'close', 'change']
    pred_data = pd.DataFrame(predictions, columns=pred_columns)

    # 生成预测数据的索引（日期）
    last_date = historical_data.index[-1]
    pred_dates = pd.date_range(last_date + timedelta(days=1), periods=len(predictions), freq='D')

    # 为预测数据设置索引
    pred_data.index = pred_dates

    # 将历史数据和预测数据连接起来
    combined_data = historical_data.append(pred_data)

    # 传递 combined_data 到模板
    # 将拼接后的数据转换为 JSON 格式
    combined_data['trade_date'] = combined_data.index
    combined_data_json = combined_data.to_json(orient="records")

    historical_data_length = len(historical_data)
    context = {
        'data': combined_data.to_json(orient='records', date_format='iso'),
        'historical_data_length': historical_data_length
    }
    return render(request, 'stockapp/stockSinglePredict.html', context)


def stock_comparison(request):
    ts_code1 = request.GET.get('ts_code1', '000001.SZ')
    ts_code2 = request.GET.get('ts_code1', '000002.SZ')
    data1 = getData(ts_code1)
    data2 = getData(ts_code2)

    chart_names1 = [data1['name'], "macd", "kdj", "wn", "bias", "boll"]
    chart_names2 = [data2['name'], "macd", "kdj", "wn", "bias", "boll"]
    # chart_names1 = [data1['name'], "MACD", "KDJ", "W&M", "BIAS", "BOLL"]
    # chart_names2 = [data2['name'], "MACD", "KDJ", "W&M", "BIAS", "BOLL"]

    row_data = df.to_dict(orient='records')
    return render(request, 'stockapp/stockComparison.html',
                  {'stocks': row_data, 'data1': data1, 'data2': data2, 'chart_names1': chart_names1,
                   'chart_names2': chart_names2})


def market_index(request):
    return render(request,'stockapp/marketIndex.html')

def market_line(request):
    sh_legend,sh_x_axis,sh_series_list = shanghai_Index()
    sz_legend,sz_x_axis,sz_series_list = shenzhen_Index()
    sh_x_axis = sh_x_axis.split(',')
    sz_x_axis = sz_x_axis.split(',')
    context = {
        'sh_legend':sh_legend ,
        'sh_x_axis': sh_x_axis,
        'sh_series_list':sh_series_list,

        'sz_legend': sz_legend,
        'sz_x_axis': sz_x_axis,
        'sz_series_list': sz_series_list,
    }

    return JsonResponse(context)

"""
需求阐述：
predictSingStock 需求参数：
我现在有股票的dataframe数据，数据示例：“,ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount
0,000001.SZ,20230424,12.48,12.5,12.03,12.1,12.5,-0.4,-3.2,1467671.92,1798133.967”。我需要
基于keras建立LSTM模型来预测其中的high,low,close,change这4列数据。lstm的训练以及预测过程中，不需要切分训练集和测试集，把所有的历史数据作为训练模型的数据，
然后根据参数n_step来迭代预测未来n_step天的数据。要求给出的代码大体分为predict(n_step)、train(trainData)两个函数。其中train负责训练、保存模型,
predict负责读取模型、按照n_step进行迭代预测。

我有股票dataframe数据，示例如下：“,ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount
0,000001.SZ,20230428,12.26,12.78,12.22,12.55,12.26,0.29,2.3654,1445254.34,1816868.033”。
在django中，我已经利用lstm在view函数predictSingleStock中预测出了high,low,close,change未来7天的值。我现在需要把历史数据和预测
出来的数据拼接起来，传到stockSingPredict.html中的echats展示。要求将high,low,close放在同一个echats中，change单独放一个echats。
预测的值和历史数据用不同的颜色显示出来,echats的横坐标是dataframe的index，也就是数据对应的日期。整个页面基于bootstrap4开发，要美观。
我的



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


django项目中，我打算把view函数market_line(request) 中获取的2组数据传递到marketIndex.html中，用两个echarts显示出来。view代码：“def market_line(request):
    sh_legend,sh_x_axis,sh_series_list = shanghai_Index()
    sz_legend,sz_x_axis,sz_series_list = shenzhen_Index()
    context = {
        'sh_legend':sh_legend ,
        'sh_x_axis': sh_x_axis,
        'sh_series_list':sh_series_list,

        'sz_legend': sz_legend,
        'sz_x_axis': sz_x_axis,
        'sz_series_list': sz_series_list,
    }
    return JsonResponse(context)”，html代码：“{% load static %}
{% load bootstrap4 %}

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Title</title>
    <script src="{% static 'js/jquery.min.js' %}"></script>

</head>
<body>

<div class="container">
    <div class="row">
        <div class="col-md-6">
            <div id="chart1" style="width: 100%; height: 400px;"></div>
        </div>
        <div class="col-md-6">
            <div id="chart2" style="width: 100%; height: 400px;"></div>
        </div>
    </div>
</div>

<script src="https://cdn.bootcdn.net/ajax/libs/echarts/5.2.1/echarts.min.js"></script>

<script type="text/javascript">
    $(document).ready(function () {
        $.ajax({
            url: "/stock/chart/marketline/",  // 这里要修改成你的路由
            type: "GET",
            dataType: "json",
            success: function (data) {
                console.log(data)
                {#var sh_legend = JSON.parse(data.sh_legend);#}
                {#var sh_x_axis = JSON.parse(data.sh_x_axis);#}
                {#var sh_series_list = JSON.parse(data.sh_series_list);#}
                {#var sz_legend = JSON.parse(data.sz_legend);#}
                {#var sz_x_axis = JSON.parse(data.sz_x_axis);#}
                {#var sz_series_list = JSON.parse(data.sz_series_list);#}

                var sh_legend = JSON.stringify(data.sh_legend);
                var sh_x_axis = JSON.stringify(data.sh_x_axis);
                var sh_series_list = JSON.stringify(data.sh_series_list);
                var sz_legend = JSON.stringify(data.sz_legend);
                var sz_x_axis = JSON.stringify(data.sz_x_axis);
                var sz_series_list = JSON.stringify(data.sz_series_list);

                var chart1 = echarts.init(document.getElementById('chart1'));
                var option1 = {
                    legend: {
                        data: sh_legend,
                        bottom: 0,
                    },
                    tooltip: {
                        trigger: 'axis'
                    },
                    xAxis: {
                        type: 'category',
                        //data: sh_x_axis.map(function (item) {
                        //  return item[0];
                        //})
                        data: sh_x_axis
                    },
                    yAxis: {
                        type: 'value'
                    },
                    series: [{
                        data: sh_series_list,
                        //data: sh_series_list.map(function (item) {
                          //  return item[1];
                        //}),
                        type: 'line',
                        smooth: true
                    }]
                };
                chart1.setOption(option1);

                var chart2 = echarts.init(document.getElementById('chart2'));
                var option2 = {
                    legend: {
                        data: sz_legend,
                        bottom: 0,
                    },
                    tooltip: {
                        trigger: 'axis'
                    },
                    xAxis: {
                        type: 'category',
                        //data: sz_x_axis.map(function (item) {
                        //  return item.split(',')[0];
                        //})
                        data: sh_x_axis

                    },
                    yAxis: {
                        type: 'value'
                    },
                    series: [{
                        data: sz_series_list,
                        //data: sz_series_list.map(function (item) {
                          //  return item.split(',')[1];
                        //}),
                        type: 'line',
                        smooth: true
                    }]
                };
                chart2.setOption(option2);
            },
            error: function () {
                alert("Failed to get data.");
            }
        });
    });
</script>

</body>
</html>” view中的数据示例：sh_legend=['open', 'close'],   sh_x_axis = '20230321,20230322,20230323,20230324',  sh_series_list = [({'name': 'open', 'type': 'line', 'stack': 'Total', 'smooth': True, 'data': [3240.838, 3263.412, 3258.779, 3282.739, 3261.823, 3256.602, 3253.621, 3243.068, 3263.406, 3277.34, 3297.183, 3302.751, 3312.481, 3331.505, 3317.077, 3316.862, 3321.273, 3326.37, 3337.06, 3379.226, 3391.354, 3367.052, 3363.998, 3300.812, 3276.403, 3255.217, 3256.227, 3283.122, 3306.482, 3350.6538]},), ({'name': 'close', 'type': 'line', 'stack': 'Total', 'smooth': True, 'data': [3255.65, 3265.747, 3286.649, 3265.653, 3251.397, 3245.376, 3240.056, 3261.248, 3272.86, 3296.396, 3312.557, 3312.628, 3327.645, 3315.357, 3313.57, 3327.181, 3318.363, 3338.152, 3385.609, 3393.331, 3370.128, 3367.031, 3301.257, 3275.413, 3264.871, 3264.1, 3285.884, 3323.274, 3350.458, 3354.1749]},)]  。现在运行，前端页面得不到想要的效果，echats显示的x和y都有问题，帮我检查
"""
