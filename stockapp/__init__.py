import pandas as pd
import tushare as ts

"""
在项目最开始运行的适合，创建如下对象，作为全局变量
pro:股票数据接口； df:所有股票的简要信息
tushare官网：https://www.tushare.pro/
"""
# api_token = '换成你自己的tushare api'
api_token = '169295a1bc79995da8eaf9630b5d7f0c868c021ba5b9ed3cc6d8df1b'
pro = ts.pro_api(api_token)

df = pd.read_csv('data/allStock.csv').drop('symbol', axis=1)
df.index.name = 'index'

# 定义一个config字典，用于存储和管理包的配置参数
config = {
    'add_list': ['open', 'close', 'change', 'vol', 'amount'],  # add_list:将ts_code.csv中的哪些信息加入allStock.csv中
    'page_num': 50,  # stockList.html中每个分页显示的数量
    'predict_stock_count': 10,  # 用allStock.csv中的多少个股票作为预测对象
    'predict_count': 200,  # 打算利用多少天的历史数据来预测
    'stock_count': 100,  # 打算预测多少支股票
    'step_in': 3,
    'n_step': 7,
}


def fun():
    # 测试函数
    import time
    startTime = time.time()
    code = '600000.SH'
    pro = ts.pro_api(api_token)
    df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
    df.to_csv('allStock.csv')
    # 获取股票数据
    # stock_df = pro.daily_basic(ts_code=code)  #这个接口积分不够，无法调用
    # stock_df = pro.daily(ts_code=code, start_date='20190120')
    # stock_df.to_csv(f'../data/{code}.csv',index=False)
    print(time.time() - startTime)

