import tushare as ts
import pandas as pd

# pro = ts.pro_api('169295a1bc79995da8eaf9630b5d7f0c868c021ba5b9ed3cc6d8df1b')
# df = pd.read_csv('data/allStock.csv').drop('symbol',axis=1)
# df.index.name = 'index'
pro,df = None,None

def fun():
    import time
    startTime = time.time()
    code = '600000.SH'
    pro = ts.pro_api('169295a1bc79995da8eaf9630b5d7f0c868c021ba5b9ed3cc6d8df1b')
    # 获取股票数据
    # stock_df = pro.daily_basic(ts_code=code, start_date='20190120')  这个接口积分不够，无法调用
    stock_df = pro.daily(ts_code=code, start_date='20190120')
    stock_df.to_csv(f'../data/{code}.csv',index=False)

    print(time.time()-startTime)

fun()





