# coding: utf-8

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

cmap_bold = ListedColormap(['b', 'g', 'r'])

class KNN:
    def load_data(self, filename):
        fr = open(filename)
        numberOfLines = len(fr.readlines())
        returnMat = np.zeros((numberOfLines, 3))  # prepare matrix to return classLabelVector = [] # prepare labels return
        classLabelVector = []  # prepare labels return
        fr = open(filename)
        index = 0
        for line in fr.readlines():
            line = line.strip()
            listFromLine = line.split('\t')
            returnMat[index, :] = listFromLine[0:3]
            classLabelVector.append(int(listFromLine[-1]))
            index += 1
        return returnMat, classLabelVector

    def draw(self, dataSetX, dataSetY):
        fig = plt.figure()
        #2行2列第一个
        ax1 = fig.add_subplot(221)
        #参数 X轴 Y轴 s：点的像素 c：标签 cmap：标签对应颜色
        ax1.scatter(dataSetX[:, 0], dataSetX[:, 1], s=5, c=dataSetY, cmap=cmap_bold)
        #2行2列第二个
        ax2 = fig.add_subplot(222)
        ax2.scatter(dataSetX[:, 0], dataSetX[:, 2], s=5, c=dataSetY, cmap=cmap_bold)
        #2行2列第三个整行
        ax3 = fig.add_subplot(212)
        ax3.scatter(dataSetX[:, 1], dataSetX[:, 2], s=5, c=dataSetY, cmap=cmap_bold)
        plt.show()


if __name__ == "__main__":
    knn = KNN()
    returnMat, classLabelVector = knn.load_data('./KNN/datingTestSet2.txt')
    knn.draw(returnMat, classLabelVector)
