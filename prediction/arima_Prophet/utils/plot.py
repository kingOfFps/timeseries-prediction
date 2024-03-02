import matplotlib.pyplot as plt

"""
    存放一些用户画图的函数：训练loss曲线、预测值与真实值对比。。。
"""

# 展示训练过程中loss的变化
def showTrainLoss(history, isShow=False):
    if not isShow:
        return
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()


# 展示预测结果和真实值
def showTrueAndPredict1(yTrue, yPredict, title=None):
    pointCount = min(min(yTrue.shape[0], yPredict.shape[0]) // 3, 50)
    aa = [x for x in range(pointCount)]
    plt.plot(aa, yTrue[:pointCount], marker='.', label="真实值")
    plt.plot(aa, yPredict[:pointCount], 'r', label="预测值")
    plt.legend()
    plt.ylabel('y', size=15)
    plt.xlabel('Time step', size=15)
    plt.title(title)
    plt.legend(fontsize=15)
    plt.show()

def showTrueAndPred2(yTrue, yPredict):
    plt.plot(yTrue, label='yTrue')
    plt.plot(yPredict, label='yPred')
    plt.legend()
    plt.show()

