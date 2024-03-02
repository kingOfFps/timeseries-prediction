

时间序列预测项目，采用不同的深度学习、机器学习算法，进行时间序列预测。包含不同预测算法与预测场景

## 依赖

- Python 3.6
- Keras
- Tensorflow 1.15
- numpy
- pandas
- matplotlib
- scipy

**数据集1：stock**

股票数据集，包含上证500多家上市公式的股票时间序列

数据各列说明：

ts_code：股票代码

trade_date：交易日期

open：开盘价

close：收盘价

high：最高价

low：最低价

change：变化率

amount：总量

**数据集2：all_season.csv**

数据集为池塘溶解氧时间序列数据，采集方式是通过采集设备每隔半小时在水面一下0.2米记录一次。

数据各列说明:

Temp：水温 ◦C

EC：导电率  µS/cm 

PH：PH值

turbidity：浊度 NTU

Chl-a：叶绿素a浓度 µg/L

DO：溶解氧含量 mg/L 

**数据集2：daily-min-temperatures.csv**

气温数据集，记录了1981/1/1到1990/12/31温度的数据集，数据频率为1天。



