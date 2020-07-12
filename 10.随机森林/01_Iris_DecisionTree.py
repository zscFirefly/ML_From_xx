import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import  Pipeline

from sklearn.datasets import load_iris

def iris_type(s):
    it = {'Iris-setos':0,'Iris-versicolor':1,'Iris-virginica':2}
    return it[s]

# 花萼长度、花萼宽度、花瓣长度、花瓣宽度
iris_feature = u'花萼长度',u'花萼宽度',u'花瓣长度',u'花瓣宽度'

if __name__ == "__main__":
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    # 数据准备
    # 导入iris数据集
    data = load_iris()

    # 获取x,y值
    # target = pd.DataFrame(data=data.target)
    target = data.target
    y = np.array(target)
    feature = pd.DataFrame(data=data.data, columns=data.feature_names)
    # 取前两列数据
    x = feature[['sepal length (cm)','sepal width (cm)']].values

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)


    # 决策树参数估计
    model = Pipeline([
        ('ss',StandardScaler()),  # 数据标准化（均值为0 方差为1），为了提升模型效果
        ('DTC',DecisionTreeClassifier(criterion='entropy',max_depth=3))
    ])
    model = model.fit(x_train,y_train)
    y_test_hat = model.predict(x_test)

    # 等价于
    # ss = StandardScaler()
    # x = ss.fit_transform(x)
    # model = DecisionTreeClassifier(criterion = 'entropy',max_depth=5)
    # model = model.fit(x_train,y_train)

    # 保存
    # dot -Tpng -o 1.pnt 1.dot
    f = open('iris_tree.dot','w')
    tree.export_graphviz(model.get_params('DTC')['DTC'],out_file = f) # DTC为上文标记的参数
    f.close()

    # 画图
    # 画图数据准备
    N,M = 100,200
    x1_min,x1_max = x[:,0].min(),x[:,0].max()
    x2_min,x2_max = x[:,1].min(),x[:,1].max()
    # 通过MN划分方格
    t1 = np.linspace(x1_min,x1_max,M)
    t2 = np.linspace(x2_min,x2_max,N)
    x1,x2 = np.meshgrid(t1,t2)
    x_show = np.stack((x1.flat,x2.flat),axis = 1) #flat 将x1、x2二维数组分别拉直 并调用stack整合x1与x2


    # 颜色准备
    cm_light = mpl.colors.ListedColormap(['#A0FFA0','#FFA0A0','#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g','r','b'])
    x1_min,x1_max = x[:,0].min(),x[:,0].max()
    # y值预测情况
    y_show_hat = model.predict(x_show)
    y_show_hat = y_show_hat.reshape(x1.shape)
    plt.figure(facecolor = 'w')
    plt.pcolormesh(x1,x2,y_show_hat,cmap= cm_light) # 预测值的显示
    plt.scatter(x_test[:,0],x_test[:,1],c=y_test.ravel(),edgecolor = 'k',s = 100,cmap=cm_dark,marker='o')
    plt.scatter(x[:,0],x[:,1],c=y.ravel(),edgecolor = 'k',s = 40,cmap=cm_dark) # 全部数据
    plt.xlabel(iris_feature[0],fontsize = 15)
    plt.ylabel(iris_feature[1],fontsize = 15)
    plt.xlim([x1_min,x1_max])
    plt.ylim([x2_min,x2_max])
    plt.title(u"鸢尾花数据的决策树分类")
    plt.grid(True) #加上网格
    plt.show()

    # 训练集上的预测结果
    y_test = y_test.reshape(-1)
    result = (y_test_hat == y_test) #True则预测正确，False则预测错误
    acc = np.mean(result)
    print("准确度：%.2f%%"%(100*acc))


    # 决策层深度的正确率
    depth = np.arange(1,15)
    err_list = []
    for d in depth:
        clf = DecisionTreeClassifier(criterion='entropy',max_depth=d)
        clf = clf.fit(x_train,y_train)
        y_test_hat = clf.predict(x_test)
        result = (y_test_hat == y_test)
        err = 1 - np.mean(result)
        err_list.append(err)
        print("决策树深度为：%d，错误率：%.2f%%"%(d ,(100 * err)))
    plt.figure(facecolor='w')
    plt.plot(depth,err_list,'ro-',lw=2)
    plt.xlabel(u'决策树深度',fontsize=15)
    plt.ylabel(u"错误率",fontsize=15)
    plt.title(u"决策树深度与过拟合",fontsize=17)
    plt.grid(True)
    plt.show()

