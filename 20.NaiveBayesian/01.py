import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
def iris_type(s):
    it = {'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}
    return it[s]

if __name__ == "__main__":
    data = load_iris()
    x = data['data']
    y = data['target']
    x = x[:,:2]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=1)

    # x_train = x_train[:,:2]
    gnb = Pipeline([
        ('sc',StandardScaler()),
        ('clf',GaussianNB())
    ])
    # gnb = KNeighborsClassifier(n_neighbors=3).fit(x,y.ravel())
    # 多项拟合
    # gnb = Pipeline([
    #     ('sc',MinMaxScaler()),
    #     ('clf',MultinomialNB())
    # ])
    gnb.fit(x_train,y_train.ravel())

    N,M = 500,500
    x1_min,x1_max = x[:,0].min(),x[:,0].max()
    x2_min,x2_max = x[:,1].min(),x[:,1].max()
    t1 = np.linspace(x1_min, x1_max, M)
    t2 = np.linspace(x2_min, x2_max, N)
    x1, x2 = np.meshgrid(t1, t2)
    x_pic = np.stack((x1.flat,x2.flat),axis=1)

    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    cm_light = mpl.colors.ListedColormap(['#A0FFA0','#FFA0A0','#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g','r','b'])
    y_hat = gnb.predict(x_pic)
    y_hat = y_hat.reshape(x1.shape)
    plt.figure(facecolor = 'w')
    plt.pcolormesh(x1,x2,y_hat,cmap=cm_light)
    plt.scatter(x[:,0],x[:,1],c=y,edgecolors='k',s=50,cmap=cm_dark)
    plt.xlabel("花萼长度",fontsize=14)
    plt.ylabel("花萼宽度",fontsize=14)
    plt.xlim(x1_min,x1_max)
    plt.ylim(x2_min,x2_max)
    plt.title("GaussianNB对鸢尾花的分类结果",fontsize=18)
    plt.grid(True)
    plt.show()

    y_test_hat = gnb.predict(x_test)
    y_test_hat = y_test_hat.reshape(-1)
    result = y_test == y_test_hat
    print(y)
    print(result)
    acc = np.mean(result)
    print("在测试集上的准确度：%.2f%%"%(100*acc))