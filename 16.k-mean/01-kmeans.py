import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import matplotlib.colors
from sklearn.cluster import KMeans

def expand(a,b):
    d = (b-a) * 0.1
    return a-d ,b+d

if __name__ == "__main__":
    N = 400
    centers = 4
    # make_blobs 聚类样本生成器；centers类别数、n_features:每个样本特征数；cluster_std：没个类别的方差
    data,y = ds.make_blobs(N,n_features=2,centers=centers,random_state=2)
    data2,y2 = ds.make_blobs(N,n_features=2,centers=centers,cluster_std=(1,2.5,0.5,2),random_state=2)
    data3 = np.vstack((data[y==0][:],data[y==1][:50],data[y==2][:20],data[y==3][:5]))
    y3 = np.array([0]*100 + [1] * 50 + [2] * 20 + [3] * 5)
    # print(y3)


    cls = KMeans(n_clusters=4,init='k-means++')
    y_hat = cls.fit_predict(data)
    y2_hat = cls.fit_predict(data2)
    y3_hat = cls.fit_predict(data3)

    m = np.array(((1,1),(1,3)))
    data_r = data.dot(m) # 对数据旋转处理
    y_r_hat = cls.fit_predict(data_r)

    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    matplotlib.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    cm = matplotlib.colors.ListedColormap(list('rgbm'))

    plt.figure(figsize=(9,10),facecolor='w')
    plt.subplot(421)
    plt.title("原始数据")
    # print(data)
    plt.scatter(data[:,0],data[:,1],c=y,s=30,cmap=cm,edgecolors='none')
    x1_min,x2_min =  np.min(data,axis=0)
    x1_max,x2_max =  np.max(data,axis=0)
    # 调整边界
    x1_min,x1_max = expand(x1_min,x1_max)
    x2_min,x2_max = expand(x2_min,x2_max)
    plt.xlim((x1_min,x2_max))
    plt.ylim((x2_min,x2_max))
    plt.grid(True)

    plt.subplot(422)
    plt.title("KMeans++聚类")
    plt.scatter(data[:,0],data[:,1],c = y_hat,s=30,cmap=cm,edgecolors='none')
    x1_min,x2_min =  np.min(data,axis=0)
    x1_max,x2_max =  np.max(data,axis=0)
    plt.grid(True)

    plt.subplot(423)
    plt.title("旋转后数据")
    plt.scatter(data_r[:,0],data_r[:,1],c = y,s=10,cmap=cm,edgecolors='none')
    x1_min, x2_min = np.min(data_r, axis=0)
    x1_max, x2_max = np.max(data_r, axis=0)
    x1_min, x1_max = expand(x1_min, x1_max)
    x2_min, x2_max = expand(x2_min, x2_max)
    plt.xlim((x1_min,x1_max))
    plt.ylim((x2_min,x2_max))
    plt.grid(True)


    plt.subplot(424)
    plt.title("旋转后KMeans++聚类")
    plt.scatter(data_r[:,0],data_r[:,1],c=y_r_hat,s=10,cmap=cm,edgecolors='none')
    plt.xlim((x1_min,x1_max))
    plt.ylim((x2_min,x2_max))
    plt.grid(True)

    plt.subplot(425)
    plt.title("方差不相等数据")
    plt.scatter(data2[:,0],data2[:,1],c=y2,s=10,cmap=cm,edgecolors='none')
    x1_min,x2_min = np.min(data2,axis=0)
    x1_max,x2_max = np.max(data2,axis=0)
    x1_min, x1_max = expand(x1_min, x1_max)
    x2_min, x2_max = expand(x2_min, x2_max)
    plt.xlim(x1_min,x1_max)
    plt.ylim(x2_min,x2_max)
    plt.grid(True)

    plt.subplot(426)
    plt.title("方差不相等KMeans++聚类")
    plt.scatter(data2[:,0],data2[:,1],c=y2_hat,s=10,cmap=cm,edgecolors='none')
    plt.xlim(x1_min,x1_max)
    plt.ylim(x2_min,x2_max)
    plt.grid(True)

    plt.subplot(427)
    plt.title("数量不相等数据")
    print(len(data3))
    print(len(y3))
    plt.scatter(data3[:, 0], data3[:, 1], c=y3, s=10, cmap=cm, edgecolors='none')
    x1_min,x2_min = np.min(data3,axis=0)
    x1_max,x2_max = np.max(data3,axis=0)
    x1_min, x1_max = expand(x1_min, x1_max)
    x2_min, x2_max = expand(x2_min, x2_max)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.grid(True)


    plt.subplot(428)
    plt.title("数量不相等KMeans++聚类")
    plt.scatter(data3[:, 0], data3[:, 1], c=y3_hat, s=10, cmap=cm, edgecolors='none')
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.grid(True)

    plt.tight_layout(2)
    plt.suptitle("数据分布对KMeans聚类的影响",fontsize=18)
    plt.subplots_adjust(top=0.92)
    plt.show()


