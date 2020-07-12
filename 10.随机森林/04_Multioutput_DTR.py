import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
if __name__ == "__main__":
    # 数据生成
    N = 300
    x = np.random.rand(N) * 8 - 4
    x.sort()
    # y1 = np.sin(x) + np.random.rand(N) * 0.05
    # y2 = np.cos(x) + np.random.rand(N) * 0.1
    y1 = 16 * np.sin(x) ** 3 + np.random.randn(N)
    y2 = 13 * np.cos(x) - 5 * np.cos(2*x) - 2 * np.cos(3*x) - np.cos(4*x) + 0.1 * np.random.randn(N)
    y = np.vstack((y1,y2)).T
    x = x.reshape(-1,1) # 转置后得到N个样本，每个样本都是一维的

    # 训练数据
    deep = 5
    reg = DecisionTreeRegressor(criterion = "mse",max_depth = deep)
    dt = reg.fit(x,y)

    #测试训练结果
    x_test = np.linspace(-4,4,num = 1000).reshape(-1,1)
    y_hat = dt.predict(x_test)

    # 绘图
    plt.scatter(y[:,0],y[:,1],c = 'r',s=40,label='Actual')
    plt.scatter(y_hat[:,0],y_hat[:,1],c='g',marker='s',s=100, label='Depth=%d'% deep,alpha=1)
    plt.legend(loc='upper left')
    plt.xlabel('y1')
    plt.ylabel('y2')
    plt.grid()
    plt.show()