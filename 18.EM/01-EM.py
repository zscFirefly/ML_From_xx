import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances_argmin


matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决保存图像是负号'-'显示为方块的问题

if __name__ == '__main__':
    style = 'sklearn'
    np.random.seed(0)
    mu1_fact = (0,0,0) # 均值在原点
    cov_fact = np.identity(3) # 方差为3
    data1 = np.random.multivariate_normal(mu1_fact,cov_fact,400)
    print(data1)
    mu2_fact = (2,2,1) # 均值在2,2,1处
    cov_fact = np.identity(3) #方差为3
    data2 = np.random.multivariate_normal(mu2_fact,cov_fact,100)
    # vstack将两组数据堆叠到一起，要分析的data
    data = np.vstack((data1,data2))
    # 产生y值，以备计算准确率
    y = np.array([True]*400+[False]*100)

    # 高斯分布
    g = GaussianMixture(n_components=2,covariance_type='full',tol=1e-6,max_iter=1000)
    # n_components:分布个数；covariance_type:分布的类型（full为任意高斯分布）；max_iter:最大迭代次数
    g.fit(data)
    print("类别概率：\t",g.weights_,'\n') #也可以理解为每个类别占比
    print("均值：\t",g.means_)
    print("方差:\t",g.covariances_,'\n')
    mu1,mu2 = g.means_
    sigma1,sigma2 = g.covariances_


    # 预测
    # 以预测出来你的结果 获得多元高斯分布的概率密度函数
    norm1 = multivariate_normal(mu1,sigma1)
    norm2 = multivariate_normal(mu2,sigma2)
    tau1 = norm1.pdf(data)# 求样本的概率密度
    tau2 = norm2.pdf(data)

    fig = plt.figure(figsize=(13,7),facecolor='w')
    ax = fig.add_subplot(121,projection='3d')
    ax.scatter(data[:,0],data[:,1],data[:,2],c='b',s=30,marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("原始数据",fontsize=18)
    ax = fig.add_subplot(122,projection='3d')
    # 判断mu1_fact,mu2_fact与mu1,mu2是否一样
    # c1与c2的结果为True or False及为y值情况
    order = pairwise_distances_argmin([mu1_fact,mu2_fact],[mu1,mu2],metric='euclidean')
    print(order)
    if order[0] == 0:
        c1 = tau1 > tau2
    else:
        c1 = tau1 < tau2
    c2 = ~c1
    # 计算True和False即准确率
    acc = np.mean(y == c1)
    print("准确率：%.2f%%"%(100*acc))
    ax.scatter(data[c1,0],data[c1,1],data[c1,2],c='r',s=30,marker='o', depthshade=True)
    ax.scatter(data[c2,0],data[c2,1],data[c2,2],c='r',s=30,marker='^',depthshade=True)
    ax.set_xlabel('X')
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("EM算法分类",fontsize=18)
    plt.tight_layout()
    plt.show()
