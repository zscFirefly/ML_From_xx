
from sklearn.datasets import load_iris
from sklearn.model_selection import  train_test_split
from sklearn import svm
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def show_accuracy(a,b,tip):
     acc = a.ravel() == b.ravel()
     t = np.mean(acc)
     print(tip+'正确率'+str(t))

def iris_type(s):
    it = {'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}
    return it[s]

iris_feature = ['花萼长度',u'花萼宽度',u'花瓣长度',u'花瓣宽度']
if __name__ == '__main__':
    iris = load_iris()
    x = iris.data
    y = iris.target
    x = x[:,:2]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state = 1)


    clf = svm.SVC(C=0.8,kernel='rbf',gamma=20,decision_function_shape='ovr')
    clf.fit(x_train,y_train.ravel())

    # 准确率
    print(clf.score(x_train,y_train)) # 精度
    y_hat = clf.predict(x_train)
    show_accuracy(y_hat,y_train,'训练集')
    print(clf.score(x_test,y_test))
    y_hat = clf.predict(x_test)
    show_accuracy(y_hat,y_test,"测试集")

    # 画图
    x1_min,x1_max = x[:,0].min(),x[:,0].max() # 第0列的范围
    x2_min,x2_max = x[:,1].min(),x[:,1].max() # 第1列的范围
    x1,x2 = np.mgrid[x1_min:x1_max:500j,x2_min:x2_max:500j] # 生成网格采样点
    grid_test = np.stack((x1.flat,x2.flat),axis=1)


    Z = clf.decision_function(grid_test) #样本到决策面的距离
    grid_hat = clf.predict(grid_test) # 预测分类值
    grid_hat = grid_hat.reshape(x1.shape)
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    cm_light = mpl.colors.ListedColormap(['#A0FFA0','#FFA0A0','#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g','r','b'])
    x1_min,x1_max = x[:,0].min(),x[:,0].max()
    x2_min,x2_max = x[:,1].min(),x[:,1].max() # 第1列的范围
    grid_test = np.stack((x1.flat,x2.flat),axis=1)
    plt.pcolormesh(x1,x2,grid_hat,cmap=cm_light)

    plt.scatter(x[:,0],x[:,1],c=y,edgecolors='k',s=50,cmap=cm_dark)
    plt.scatter(x_test[:,0],x_test[:,1],s=120,facecolors='none',zorder=10)
    plt.xlabel(iris_feature[0],fontsize=13)
    plt.ylabel(iris_feature[1],fontsize=13)
    plt.xlim(x1_min,x1_max)
    plt.ylim(x2_min,x2_max)
    plt.title("鸢尾花SVM二特征分类",fontsize=15)
    plt.grid()
    plt.show()




