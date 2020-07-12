import numpy as np
from sklearn import svm
import matplotlib as mpl
import matplotlib.pyplot as plt

def show_accuracy(a,b):
     acc = a.ravel() == b.ravel()
     print('正确率： %.2f%%' % (100*float(acc.sum())/a.size))


if __name__ == '__main__':
    data = np.loadtxt('./dataset/14.bipartition.txt',dtype=np.float,delimiter = '\t')
    x,y = np.split(data,(2,),axis=1)
    y[y==0] =-1
    y = y.ravel()

    # 分类器
    clfs = [svm.SVC(C=0.3,kernel='linear'),
            svm.SVC(C=10,kernel='linear'),
            svm.SVC(C=5,kernel='rbf',gamma=1),
            svm.SVC(C=5,kernel='rbf',gamma=4)]
    titles = ["Linear,C=0.3","Linear,C=10","RBF,gamma=1","RBF,gamma=4"]
    x1_min,x1_max = x[:,0].min(),x[:,0].max()
    x2_min,x2_max = x[:,1].min(),x[:,1].max()
    x1,x2 = np.mgrid[x1_min:x1_max:500j,x2_min:x2_max:500j]
    grid_test = np.stack((x1.flat,x2.flat),axis=1)

    cm_light = mpl.colors.ListedColormap(['#77E0A0','#FF8080'])
    cm_dark = mpl.colors.ListedColormap(['g','r'])
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10,8),facecolor='w')
    for i,clf in enumerate(clfs):
        clf.fit(x,y)

        y_hat = clf.predict(x)
        show_accuracy(y_hat,y)

        # 画图
        print("支持向量的数目：",clf.n_support_)
        print("支撑向量的系数：",clf.dual_coef_)
        print("支撑向量：",clf.n_support_)

        plt.subplot(2,2,i+1)
        grid_hat = clf.predict(grid_test)
        grid_hat = grid_hat.reshape(x1.shape)
        plt.pcolormesh(x1,x2,grid_hat,cmap=cm_light,alpha=0.8) # alpha 透明度
        plt.scatter(x[:,0],x[:,1],c=y,edgecolors='k',s=40,cmap=cm_dark)
        plt.scatter(x[clf.support_,0],x[clf.support_,1],edgecolors='k',facecolor='none',s=100,marker='o')# 画支撑向量
        z = clf.decision_function(grid_test) #z为到超平面的距离
        print("Z的长度：",len(z))
        z = z.reshape(x1.shape)
        plt.contour(x1,x2,z,colors=list('krk'),linestyles=['--','-','--'],linewidth=[1,2,1],levels=[-1,0,1]) # level为所画线到超平面的的距离，当值为-1、0、1时才画出来
        plt.xlim(x1_min,x1_max)
        plt.ylim(x2_min,x2_max)
        plt.title(titles[i])
        plt.grid()

    plt.suptitle(u'SVM不同参数的分类',fontsize=18)
    plt.tight_layout(2)
    plt.subplots_adjust(top=0.92)
    plt.show()





