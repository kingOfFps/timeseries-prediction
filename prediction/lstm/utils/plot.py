import matplotlib.pyplot as plt
import pandas as pd
from utils import *

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


def getData(filePath: str, ration=0.3):
    if filePath.endswith('.csv'):
        data = pd.read_csv(filePath)
    elif filePath.endswith('.xlsx') or filePath.endswith('.xls'):
        data = pd.read_excel(filePath)
    else:
        raise Exception('请输入csv文件或者excel文件')
    # 删除时间列Timestamp
    data.drop(data.columns[0], axis=1, inplace=True)
    count = int(data.shape[0] * ration)
    # 数据过大，为了节省时间，只部分数据
    return data.iloc[:count, :]


"""
下面的函数，大多都是用于生成论文中图片的画图函数
"""


def showOriginalData(count):
    """展示原始数据(从采样后的数据)分布"""
    df = getData("../../data/agriculture_load_h.csv").iloc[:count, -1]
    plt.plot(df)
    plt.xlabel('Time/h', fontsize=14)
    plt.ylabel('Load/W', fontsize=14)
    # plt.ylim(0, None)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()


def showPCC(imf, title):
    import seaborn as sns
    count = 6
    data = []
    df = pd.DataFrame()
    cols = ['t']
    for i in range(1, count):
        cols.append(f't+{i}')
    for i, name in zip(range(count), cols):
        # data.append(imf[i:-count])
        df[name] = imf[i:-count + i]
    fig, ax = plt.subplots(figsize=(9, 9))
    plt.title(title, fontsize=60)
    tar = df.corr()
    # 二维的数组的热力图，横轴和数轴的ticklabels要加上去的话，既可以通过将array转换成有column
    # 和index的DataFrame直接绘图生成，也可以后续再加上去。后面加上去的话，更灵活，包括可设置labels大小方向等。
    sns.set(font_scale=2)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=20)
    sns.heatmap(tar,
                annot=True, vmax=1, vmin=0, xticklabels=True, yticklabels=True, square=True, cmap="YlGnBu")
    plt.title(title)
    plt.show()

    # ax.set_title('二维数组热力图', fontsize = 18)


def showIMFs(imfs):
    count = 360
    # 设定整个画布的尺寸
    fig = plt.figure(figsize=(10, 12))
    fontsize = 16
    for i, imf in enumerate(imfs):
        print(i)
        # ax = fig.add_subplot(len(imfs), 1, i+1)
        plt.subplot(len(imfs), 1, i + 1)
        # ax.imshow(imf[:count])
        plt.plot(imf[:count])
        if i == len(imfs) - 1:
            plt.ylabel('res', fontsize=fontsize)
        else:
            plt.ylabel(f'IMF{i + 1}', fontsize=fontsize)
        plt.xticks(fontsize=fontsize - 2)
        plt.yticks(fontsize=fontsize - 2)
    plt.subplots_adjust(hspace=0.5)  # 调整子图之间的垂直间距
    plt.xlabel('Time/h', fontsize=fontsize)
    plt.show()


def plotIMFs(imfs):
    # 画出imfs
    plt.rcParams["figure.figsize"] = (4, 2)
    for i, trainImf in enumerate(imfs):
        plt.plot(trainImf[:200])
        plt.xticks([])
        plt.yticks([])
        plt.show()
    showIMFs(imfs)


def plotPCC(imfs):
    # 画出PCC热力矩阵图
    for i, imf in enumerate(imfs):
        if i == len(imfs) - 1:
            title = 'res'
        elif i == 1:
            title = 'Original data'
        else:
            # title = f'IMF{i + 1}'
            title = f'IMF{i+2}'
        showPCC(imf, title)


def showAllModelResult_old():
    """将所有结果画到一个图中"""
    # matplotlib.use("Agg")
    data = pd.read_excel('../result/summaryOfResults.xlsx')
    # 300
    start = 360
    loss = 0
    count = 120 - loss
    flag = 0
    linestyle_list = ['-.', ':', ' ']
    for colName in data.columns:
        if colName == 'TRUE':
            # plt.plot(data[colName][:count],label = colName,linewidth=0.8)
            plt.plot(data[colName][start:start + count], 'b', label=colName, linewidth=0.8)  # 实线
        elif colName == 'EMD-BiLSTM-DLSTM':
            plt.plot(data[colName][start:start + count], 'r', label=colName, linestyle='--', linewidth=0.8)
        else:
            # continue
            # plt.plot(data[colName][:count],label = colName, linestyle='--',linewidth=0.4)

            plt.plot(data[colName][start:start + count], label=colName, linestyle=linestyle_list[flag], linewidth=0.4)
            flag += 1

    plt.xlabel('t/h')
    plt.ylabel('有功功率/w')
    # plt.legend(prop={'size': 6})
    # plt.legend(prop={'size': 6})
    plt.xlim(start, start + count + loss)
    plt.ylim(0, None)
    plt.gca().set_aspect(18)
    # plt.set(xlim=[0, 10], ylim=[0, 20], aspect=1)
    # plt.figure(figsize=(20, 20))
    # plt.show()
    plt.savefig("figure.png", dpi=300, bbox_inches='tight')


def showAllModelResult():
    """将所有结果画到一个图中"""
    # matplotlib.use("Agg")
    data = pd.read_excel('../result/summaryOfResults.xlsx')
    count = 140
    flag = 0
    linestyle_list = ['-.', ':', ' ']
    for colName in data.columns:
        if colName == 'True' or colName == '真实值':
            plt.plot(data[colName][:count], 'b', label=colName, linewidth=0.8)  # 实线
        elif colName == 'EMD-BiLSTM-DLSTM':
            plt.plot(data[colName][:count], 'r', label=colName, linestyle='--', linewidth=0.8)
        else:
            plt.plot(data[colName][:count], label=colName, linestyle='--', linewidth=0.5)

            # if flag < len(linestyle_list):
            #     plt.plot(data[colName][:count], label=colName, linestyle=linestyle_list[flag % 3], linewidth=0.5)
            # else:
            #     plt.plot(data[colName][:count], label=colName, linestyle='--', linewidth=0.5)
            # flag += 1
    plt.xlabel('时间/h')
    plt.ylabel('负荷/W')
    # plt.legend(prop={'size': 8},bbox_to_anchor=(1.05, 1))
    plt.legend(prop={'size': 7})
    plt.savefig("figure.svg", dpi=600, bbox_inches='tight')
    # plt.show()


def showAllModelResultShape():
    """将所有结果画到一个图中,利用形状曲风"""
    # matplotlib.use("Agg")
    data = pd.read_excel('../result/summaryOfResults.xlsx')
    count = 70
    flag = 0
    linestyle_list = ['-.', ':', ' ']
    length = len(linestyle_list)
    linewidth1 = 1
    linewidth2 = 1
    markersize = 4
    for colName in data.columns:
        if colName == 'True' or colName == '真实值':
            # plt.plot(data[colName][:count], label=colName, linewidth=linewidth1, linestyle='solid')  # 实线
            plt.plot(data[colName][:count], label='True', linewidth=linewidth1, linestyle='solid')  # 实线
        elif colName == 'EMD-BiLSTM-DLSTM':
            plt.plot(data[colName][:count], label=colName, linestyle='--', linewidth=linewidth1, marker='.', markersize=markersize)
        else:
            """
            ValueError: 'dashe' is not a valid value for ls; supported values are '-', '--', '-.', ':', 
            'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
            """
            if flag == 0:
                plt.plot(data[colName][:count], label=colName, linestyle=':', linewidth=linewidth2)
            elif flag == 1:
                plt.plot(data[colName][:count], label=colName, linestyle='-.', linewidth=linewidth1, marker='x', markersize=markersize)
            if flag == 3:
                plt.plot(data[colName][:count], label=colName, linestyle='dashdot', linewidth=linewidth2)
            elif flag == 4:
                plt.plot(data[colName][:count], label=colName, linestyle=':', linewidth=linewidth1, marker='*', markersize=markersize)
            if flag == 5:
                plt.plot(data[colName][:count], label=colName, linestyle='dashed', linewidth=linewidth2)
            elif flag == 6:
                plt.plot(data[colName][:count], label=colName, linewidth=linewidth1, marker='+', markersize=markersize)
            if flag == 7:
                plt.plot(data[colName][:count], label=colName, linewidth=linewidth1, marker='^', markersize=markersize)
            elif flag == 8:
                plt.plot(data[colName][:count], label=colName, linewidth=linewidth1, marker='o', markersize=markersize)
            # if flag%2 == 0:
            #     plt.plot(data[colName][:count], label=colName, linestyle=linestyle_list[flag % length], linewidth=0.5)
            # else:
            #     plt.plot(data[colName][:count], label=colName, linestyle=linestyle_list[flag % length],marker='+', linewidth=0.2)
            flag += 1


    plt.xlabel('Time/h')
    plt.ylabel('Load/W')
    # plt.legend(prop={'size': 8},bbox_to_anchor=(1.05, 1))
    plt.legend(prop={'size': 9})
    # 绘制灰度图
    # 将其显示为灰度图
    # plt.imshow(data, cmap='gray')
    plt.savefig("figure.svg", dpi=600, bbox_inches='tight',cmap='gray')
    # plt.show()


if __name__ == '__main__':
    init()
    showOriginalData(24*15)
    # showAllModelResultShape()

"""




"""
