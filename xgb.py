from utils import *
from dataProcessing import *
from xgboost import XGBRegressor


def train(train_data, stepIn, stepOut):
    """将普通的原始数据变为监督学习格式的数据"""
    train_X, train_y = prepare_y(train_data, stepIn, stepOut)
    train_X = train_X.reshape(train_X.shape[0], -1)
    train_y = train_y.reshape(-1)
    from xgboost import XGBRegressor
    model = XGBRegressor()
    model.fit(train_X, train_y)
    return model


# 模型预测函数
def predict(X, stepIn, stepOut):
    """测试数据X需要和训练数据做相同的滑动窗口切分处理，只不过训练集有标签y，真实场景下的预测没有y"""
    X, _ = prepare_data(X, np.array([]), stepIn, stepOut)
    X = X.reshape(X.shape[0], -1)
    pred = model.predict(X)
    return pred.reshape(pred.shape[0], -1)


"""配置各种参数
stepIn:相当于一次用多少个样本来预测，也就是滑动窗口的大小
stepOut: 相当于一次预测多少步y。目前的代码逻辑只能选择stepOut=1
"""
params = {'epoch': 30, 'units': 50, 'stepIn': 2, 'stepOut': 1, 'ration': 0.8}

if __name__ == "__main__":
    init()
    data = getData('data/load.csv')[['actual']]
    """切分数据集"""
    trainXy, testXy = splitTrainTest(data, params['ration'])
    # 归一化处理
    trainy, testy, scaler = spliteAndNormalizeY(trainXy, testXy)
    # 训练测试
    model = train(trainy, params['stepIn'], params['stepOut'])
    pred = predict(testy, params['stepIn'], params['stepOut'])
    testy = testy[:pred.shape[0], ]
    # 评估
    # evaluate(testy,pred)
    # 反归一化，使得数据变回原来的量纲
    pred = scaler.inverse_transform(pred)
    testy = scaler.inverse_transform(testy)

    """因为滑动窗口切分的原因，导致预测出来的pred大小和testData的大小不一样，
    所以这里进行处理，保证大小一样，方便后面画图。
    注意：这里切片用的是[-pred.shape[0]:, -1]，而不是[:pred.shape[0], -1]
    """
    # 画图对比测试集和预测结果
    showTruePred(testy, pred)
    print(params)
    evaluate(testy, pred)

"""
运行结果记录
(20759, 77)
{'epoch': 30, 'units': 50, 'stepIn': 2, 'stepOut': 1, 'ration': 0.8}
Test RMSE: 507.08
Test MAPE: 4.49
皮尔森系数0.86
决定系数0.73
"""
