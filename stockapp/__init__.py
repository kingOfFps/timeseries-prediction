import tushare as ts
import pandas as pd

pro = ts.pro_api('169295a1bc79995da8eaf9630b5d7f0c868c021ba5b9ed3cc6d8df1b')
stock_df = pd.read_csv('data/allStock.csv').drop('symbol',axis=1)
stock_df.index.name = 'index'

def fun():
    import time
    startTime = time.time()
    code = '000001.SZ'
    # 获取股票数据
    # stock_df = pro.daily_basic(ts_code=code, start_date='20190120')  这个接口积分不够，无法调用
    stock_df = pro.daily(ts_code=code, start_date='20190120')
    stock_df.to_csv(f'../data/{code}.csv')

    print(time.time()-startTime)




