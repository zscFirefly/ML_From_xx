
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import sklearn.datasets as ds
from sklearn.preprocessing import StandardScaler

def expand(a,b):
    d = (b-a) * 0.1
    return a-d ,b+d


if __name__ == "__main__":
    # N = 1000
    # centers = [[1,2],[-1,1],[1,-1],[-1,1]]
    # data,y = ds.make_blobs(N,n_features=2,centers=centers,cluster_std=[0.5,0.25,0.7,0.5],random_state=0)
    # data = StandardScaler().fit_transform(data)
    # params = ((0.2,5),(0.2,10),(0.2,15),(0.3,5),(0.3,10),(0.3,15))

    t = np.arange(0,2*np.pi,0.1)
    data1 = np.vstack((np.cos(t),np.sin(t))).T
    data2 = np.vstack((2*np.cos(t),2*np.sin(t))).T
    data3 = np.vstack((3*np.cos(t),3*np.sin(t))).T
    data = np.vstack((data1,data2,data3))
    # 数据2的参数
    params = ((0.5,3),(0.5,5),(0.5,10),(1.,3),(1.,10),(1.,20))

    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    matplotlib.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题\
    cm = matplotlib.colors.ListedColormap(list('rgbm'))
    plt.figure(figsize=(12,8),facecolor='w')
    plt.suptitle("DBSACN聚类",fontsize=20)

    for i in range(6):
        eps, min_samples = params[i]
        model = DBSCAN(eps=eps, min_samples=min_samples) # 参数eps：为半径，min_samples：为半径内的样本数
        model.fit(data)
        y_hat = model.labels_ # 每个类别的属性，-1为噪声

        core_indices = np.zeros_like(y_hat, dtype=bool)
        core_indices[model.core_sample_indices_] = True #core_sample_indices_ 找出核心的样本点
        y_unique = np.unique(y_hat)
        n_clusters = y_unique.size - (1 if -1 in y_hat else 0)
        print (y_unique, '聚类簇的个数为：', n_clusters)
    #
        # plt.scatter(data[:, 0], data[:, 1], c=y_hat, s=15, cmap=cm, edgecolors='none')
        plt.subplot(2, 3, i + 1)
        clrs = plt.cm.Spectral(np.linspace(0, 0.8, y_unique.size))
        print(clrs)
        for k, clr in zip(y_unique, clrs):
            cur = (y_hat == k)
            if k == -1:
                plt.scatter(data[cur, 0], data[cur, 1], s=20, c='k')
                continue
            plt.scatter(data[cur, 0], data[cur, 1], s=30, c=clr, edgecolors='k') # 画点
            plt.scatter(data[cur & core_indices][:, 0], data[cur & core_indices][:, 1], s=10, c=clr, marker='o', # 核心样本点
                        edgecolors='k')
        x1_min, x2_min = np.min(data, axis=0)
        x1_max, x2_max = np.max(data, axis=0)
        x1_min, x1_max = expand(x1_min, x1_max)
        x2_min, x2_max = expand(x2_min, x2_max)
        plt.xlim((x1_min, x1_max))
        plt.ylim((x2_min, x2_max))
        plt.grid(True)
        plt.title('$\epsilon$ = %.1f  m = %d，聚类数目：%d' % (eps, min_samples, n_clusters), fontsize=16)
    plt.tight_layout(1.2)
    plt.subplots_adjust(top=0.92)
    plt.show()
