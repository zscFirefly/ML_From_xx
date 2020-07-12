
import numpy as np
from sklearn import svm
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import warnings

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(0)

    c1 =990
    c2 =10
    N = c1+c2
    x_c1 = 3*np.random.randn(c1,2)
    x_c2 = 0.5*np.random.randn(c2,2)+(4,4)
    x = np.vstack((x_c1,x_c2))
    y = np.ones(N)
    y[:c1] = -1

    # 显示大小
    s = np.ones(N) * 30
    s[:c1] = 10

    # 分类器
    clfs = [svm.SVC(C=1,kernel='linear'),
            svm.SVC(C=1,kernel='linear',class_weight={-1:1,1:10}),
            svm.SVC(C=0.8,kernel='rbf',gamma=0.5,class_weight={-1:1,1:2}),
            svm.SVC(C=0.8,kernel='rbf',gamma=0.5,class_weight={-1:1,1:10})]

    titles = ["Linear","Linear","Weight=50","RBF,Weight=2",'RBF,Weight=10']
    x1_min,x1_max = x[:,0].min(),x[:,0].max()
    x2_min,x2_max = x[:,1].min(),x[:,1].max()
    x1,x2 = np.mgrid[x1_min:x1_max:500j,x2_min:x2_max:500j]
    grid_test = np.stack((x1.flat,x2.flat),axis=1)
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    cm_light = mpl.colors.ListedColormap(['#77E0A0','#FF8080'])
    cm_dark = mpl.colors.ListedColormap(['g','r'])
    plt.figure(figsize=(10,8),facecolor='w')
    for i,clf in enumerate(clfs):
        clf.fit(x,y)

        y_hat = clf.predict(x)

        print(i+1,"次")
        print("正确率:\t",accuracy_score(y,y_hat))
        print("精度:\t",precision_score(y,y_hat,pos_label=1))
        print("召回率:\t",recall_score(y,y_hat,pos_label=1))
        print("F1-Score:\t",f1_score(y,y_hat,pos_label=1))


        plt.subplot(2,2,i+1)
        grid_hat = clf.predict(grid_test)
        grid_hat = grid_hat.reshape(x1.shape)
        plt.pcolormesh(x1,x2,grid_hat,cmap=cm_light,alpha=0.8) # alpha 透明度
        plt.scatter(x[:,0],x[:,1],c=y,edgecolors='none',s=s,cmap=cm_dark)
        # plt.scatter(x[clf.support_,0],x[clf.support_,1],edgecolors='k',facecolor='none',s=5,marker='o')# 画支撑向量
        z = clf.decision_function(grid_test) #z为到超平面的距离
        z = z.reshape(x1.shape)
        plt.contour(x1,x2,z,colors=list('krk'),linestyles=['--','-','--'],linewidth=[1,2,1],levels=[-1,0,1]) # level为所画线到超平面的的距离，当值为-1、0、1时才画出来
        plt.xlim(x1_min,x1_max)
        plt.ylim(x2_min,x2_max)
        plt.title(titles[i])
        plt.grid()
    plt.suptitle(u'不平衡数据的处理', fontsize=18)
    plt.tight_layout(2)
    plt.subplots_adjust(top=0.92)
    plt.show()


