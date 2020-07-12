import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import pandas as pd

if __name__ == "__main__":
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    # 导数据
    data = load_iris()
    target = data.target
    y = np.array(target)
    feature = pd.DataFrame(data=data.data, columns=data.feature_names).values
    feature_names = data.feature_names
    # 取前两列数据
    # x = feature[['sepal length (cm)', 'sepal width (cm)']].values
    feature_pairs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    plt.figure(figsize=(10,9),facecolor='w')
    for i,pair in enumerate(feature_pairs):
        # 准备数据
        x = feature[:,pair]

        clf = RandomForestClassifier(n_estimators=50,criterion='entropy',max_depth=4)
        dt_clf = clf.fit(x,y.ravel())

        N, M = 500, 500
        x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
        x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
        # 通过MN划分方格
        t1 = np.linspace(x1_min, x1_max, M)
        t2 = np.linspace(x2_min, x2_max, N)
        x1, x2 = np.meshgrid(t1, t2)
        x_test = np.stack((x1.flat, x2.flat), axis=1)  # flat 将x1、x2二维数组分别拉直 并调用stack整合x1与x2

        # 在训练集上预测结果
        y_hat = dt_clf.predict(x)
        y = y.reshape(-1)
        c = np.count_nonzero(y_hat == y)
        print("特征：",feature_names[pair[0]] + "+" + feature_names[pair[1]])
        print("\t预测正确数目：", c)
        print("\t准确率：%.2f%%" % (100 * float(c) / float(len(y))))

        # 显示
        cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
        cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
        x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
        # 作图图上每个值的情况
        y_show_hat = dt_clf.predict(x_test)
        y_show_hat = y_show_hat.reshape(x1.shape)
        plt.subplot(2, 3, i + 1)
        plt.pcolormesh(x1, x2, y_show_hat, cmap=cm_light)  # 预测值的显示
        plt.scatter(x[:, 0], x[:, 1], c=y.ravel(), s=10, cmap=cm_dark)  # 全部数据
        plt.xlabel(feature_names[pair[0]], fontsize=15)
        plt.ylabel(feature_names[pair[1]], fontsize=15)
        plt.xlim([x1_min, x1_max])
        plt.ylim([x2_min, x2_max])
        plt.grid()
    plt.tight_layout(2.5)
    plt.subplots_adjust(top=0.92)
    plt.suptitle(u"随机森林对鸢尾花数据的两特征祝贺的分类结果",fontsize = 18)
    plt.show()
