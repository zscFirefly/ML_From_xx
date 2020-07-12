import numpy as np
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from time import time
from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib as mpl

def test_clf(clf):
    print("分类器",clf)
    alpha_can = np.logspace(-3,2,10)
    model = GridSearchCV(clf,param_grid={'alpha':alpha_can},cv=5)
    m = alpha_can.size
    if hasattr(clf,'alpha'):
        model.set_params(param_grid={'alpha':alpha_can})
        m = alpha_can.size
    if hasattr(clf,'n_neighbors'):
        neighbors_can = np.arange(1,15)
        model.set_params(param_grid={'n_neighbors':neighbors_can})
        m = neighbors_can.size
    if hasattr(clf,'C'):
        C_can = np.logspace(1,3,3)
        gamma_can = np.logspace(-3,0,3)
        model.set_params(param_grid={'C':C_can,'gamma':gamma_can})
        m = C_can.size * gamma_can.size
    if hasattr(clf,'max_depth'):
        max_depth_can = np.arange(4,10)
        model.set_params(param_grid={'max_depth':max_depth_can})
        m = max_depth_can.size
    t_start = time()
    model.fit(x_train,y_train)
    t_end = time()
    t_train = (t_end - t_start) / (5*m)
    print("5折交叉验证的训练时间为：%.3f秒/（5*%d）=%.3f秒" % ((t_end - t_start),m,t_train))
    print("最优超参数为：",model.best_params_)
    t_start = time()
    y_hat = model.predict(x_test)
    print(y_hat)
    t_end = time()
    t_test = t_end - t_start
    print("测试时间：%.3f秒"  % t_test)
    acc = metrics.accuracy_score(y_test,y_hat)
    print("测试集准确率：%.2f%%" % (100 * acc))
    name = str(clf).split('(')[0]
    index = name.find('Classifier')
    if index != -1:
        name = name[:index]
    if name == "SVC":
        name = "SVM"
    return t_train,t_test,1-acc,name


if __name__ == "__main__":
    print("开始下载/加载数据....")
    t_start = time()
    remove = ()
    categories = "alt.atheism","talk.religion.misc","comp.graphics","sci.space"
    data_train = fetch_20newsgroups(subset="train",categories=categories,shuffle=True,random_state=0,remove=remove)
    data_test = fetch_20newsgroups(subset="test",categories=categories,shuffle=True,random_state=0,remove=remove)
    t_end = time()
    print("下载/加载数据完成，耗时%.3f秒" % (t_end - t_start))
    print("数据类型：",type(data_train))
    print("训练集包含的文本数目：",len(data_train.data))
    print("测试集包含的文本数目：",len(data_test.data))
    print("训练集和测试集使用的%d个类别的名称" % len(categories))
    categories = data_train.target_names
    pprint(categories)   # 文本类别
    y_train = data_train.target
    y_test = data_test.target  # 目标分类 转化成[0,1,2,3]
    print("--前10个文本--")
    for i in np.arange(10):
        print("文本%d(属于类别 - %s)：" % (i+1,categories[y_train[i]]))
        print(data_train.data[i])
        print("\n\n\n")

    vectorizer = TfidfVectorizer(input='content',stop_words='english',max_df=0.5,sublinear_tf=True) # 设置模型参数
    x_train = vectorizer.fit_transform(data_train.data)
    x_test = vectorizer.transform(data_test.data)
    print("训练集样本个数：%d，特征个数：%d" % x_train.shape)
    print("停用词:\n")
    pprint(vectorizer.get_stop_words())
    feature_names = np.asarray(vectorizer.get_feature_names())

    print("\n\n================\n分类器的比较：\n")
    clfs = (
        MultinomialNB(),
        BernoulliNB()
        # KNeighborsClassifier()
        # RidgeClassifier()
        # RandomForestClassifier(n_estimators=200)
        # SVC()
    )
    result = []
    for clf in clfs:
        a = test_clf(clf)
        result.append(a)
        print('\n')
    result = np.array(result)
    time_train,time_test,err,names = result.T
    x = np.arange(len(time_train))

    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10,7),facecolor='w')
    ax = plt.axes()
    errs = [float(i) for i in err]
    time_trains = [float(i) for i in time_train]
    time_tests = [float(i) for i in time_test]
    print(x,errs,time_train)
    b1 = ax.bar(x,errs,width=0.25,color='#77E0A0')
    ax_t = ax.twinx()
    b2 = ax_t.bar(x+0.25,time_trains,width=0.25,color='#FFA0A0')
    b3 = ax_t.bar(x+0.5,time_tests,width=0.25,color='#FF8080')
    plt.xticks(x+0.5,names,fontsize=10)
    leg = plt.legend([b1[0],b2[0],b3[0]],("错误率","训练时间","测试时间"),loc='upper left',shadow=True)
    plt.title("新闻组文本数据不同分类器间的比较",fontsize=18)
    plt.xlabel("分类器名称")
    plt.grid(True)
    plt.tight_layout(2)
    plt.show()



