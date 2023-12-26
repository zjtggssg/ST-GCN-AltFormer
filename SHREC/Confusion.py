import matplotlib
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
np.seterr(divide='ignore',invalid='ignore')

plt.rcParams['figure.figsize'] = (19.2,10.8)
# plt.rcParams['figure.figsize'] = (20.4,10.2)
filename = "/data/zjt/handgesture/pic/"

def text_save(filename, data):  # filename为写入txt文件的路径，data为要写入数据列表.

    file = os.path.join(filename, "data.txt")
    # print(file)
    data = np.array(data)
    np.savetxt(file, data, fmt="%s", delimiter=",")


class DrawConfusionMatrix:
    def __init__(self, labels_name, style,val_cc,normalize=True):
        """
		normalize：是否设元素为百分比形式
        """
        self.normalize = normalize
        self.labels_name = labels_name
        self.num_classes = len(labels_name)
        self.val_cc =val_cc
        self.style = style
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype="float32")

    def update(self, predicts, labels):
        """

        :param predicts: 一维预测向量，eg：array([0,5,1,6,3,...],dtype=int64)
        :param labels:   一维标签向量：eg：array([0,5,0,6,2,...],dtype=int64)
        :return:
        """
        for predict, label in zip(predicts, labels):
            self.matrix[predict, label] += 1

    def getMatrix(self,normalize=True):
        """
        根据传入的normalize判断要进行percent的转换，
        如果normalize为True，则矩阵元素转换为百分比形式，
        如果normalize为False，则矩阵元素就为数量
        Returns:返回一个以百分比或者数量为元素的矩阵

        """
        if normalize:
            per_sum = self.matrix.sum(axis=1)  # 计算每行的和，用于百分比计算
            for i in range(self.num_classes):
                self.matrix[i] =(self.matrix[i] / per_sum[i])   # 百分比转换
            self.matrix=np.around(self.matrix, 2)
            # self.matrix = self.matrix * 100# 保留2位小数点
            self.matrix[np.isnan(self.matrix)] = 0  # 可能存在NaN，将其设为0
        return self.matrix

    def drawMatrix(self):
        self.matrix = self.getMatrix(self.normalize)
        # plt.subplots(figsize=(30, 20))
        # # plt.figure(figsize=(10, 7))
        text_save(filename, self.matrix)
        plt.imshow(self.matrix, cmap=plt.cm.Blues)  # 仅画出颜色格子，没有值
        # sns.heatmap(self.matrix, annot=True, cbar=None, cmap="Blues")
        plt.title("Normalized confusion matrix")  # title
        plt.xlabel("Predict label")
        plt.ylabel("Truth label")
        plt.yticks(range(self.num_classes), self.labels_name)  # y轴标签
        plt.xticks(range(self.num_classes), self.labels_name, rotation=90)  # x轴标签
        plt.xlabel('Predict label\naccuracy={:0.4f}'.format(self.val_cc))
        thresh = self.matrix.max() / 2.
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # value = float(format('%.2f' % self.matrix[y, x]))  # 数值处理
                value = format('%.2f' % self.matrix[y, x])
                plt.text(x, y, value, verticalalignment='center', horizontalalignment='center',color='white'if self.matrix[y, x] > thresh else "black")  # 写值

        plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域

        # plt.colorbar()  # 色条
        plt.savefig('/data/zjt/handgesture/pic/ConfusionMatrix_{}.png'.format(self.style), bbox_inches='tight')  # bbox_inches='tight'可确保标签信息显示全
        plt.show()
        plt.cla()