import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import svm
import os
from PIL import Image

def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    t = np.mean(acc)
    print(tip + '正确率' + str(t))

def save_image(im,i):
    im *= 15.9375
    im = 255-im
    a = im.astype(np.uint8)
    output_path = ".\\HandWritten"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    Image.fromarray(a).save(output_path + ('\\%d.png'%i))

if __name__ == "__main__":
    print("Load Train Data Start ......")
    data = np.loadtxt("./dataset/14.optdigits.tra",dtype=np.float,delimiter=',')
    x,y = np.split(data,(-1,),axis=1)
    images = x.reshape(-1,8,8) #x是3823,64 于是将64变成 8*8
    y = y.ravel().astype(np.int)

    print("Load Test Data Start...")
    data = np.loadtxt("./dataset/14.optdigits.tes", dtype=np.float, delimiter=',')
    x_test, y_test = np.split(data, (-1,), axis=1)
    images_test = x_test.reshape(-1, 8, 8)
    y_test = y_test.ravel().astype(np.int)
    print("Load Data OK...")


    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(15,9),facecolor='w')
    for index,image in enumerate(images[:16]):
        plt.subplot(4,8,index+1)
        plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
        plt.title("训练图片:%i"%y[index])

    for index,image in enumerate(images_test[:16]):
        plt.subplot(4,8,index+17)
        plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
        save_image(image.copy(),index)
        plt.title("测试图片:%i"%y[index])
    plt.tight_layout()

    clf = svm.SVC(C=1,kernel='rbf',gamma=0.001)
    print("start Learning...")
    clf.fit(x,y)
    print("Learning is OK....")
    y_hat = clf.predict(x)
    show_accuracy(y,y_hat,'训练集')
    y_hat = clf.predict(x_test)
    show_accuracy(y_test,y_hat,'测试集')

    err_images = images_test[y_test != y_hat]
    err_y_hat = y_hat[y_test != y_hat]
    err_y = y_test[y_test != y_hat]
    print("错分的预测值：",err_y_hat)
    print("错分的值：",err_y)
    plt.figure(figsize=(10,8),facecolor='w')
    for index,image in enumerate(err_images):
        if index>=12:
            break
        plt.subplot(3,4,index +1)
        plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
        plt.title("错分为：%i,真实值：%i" % (err_y_hat[index],err_y[index]))

    plt.tight_layout()
    plt.show()

