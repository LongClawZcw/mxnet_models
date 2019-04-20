# Softmax 回归跟线性回归⼀样将输⼊特征与权重做线性叠加 不同在于softmax 回归的输出值个数等于标签⾥的类别数
# 将输出值 oi 当做预测类别是 i 的置信度，并将值最⼤的输出所对应的类作为预测输出
#Softmax 运算将输出值变换成值为正且和为1的概率分布

#交叉熵损失函数 衡量两个概率分布差异的测量函数 交叉熵只关⼼对正确类别的预测概率，因为只要其值⾜够⼤，我们就可以确保分类结果正确
#最小化交叉熵损失函数等价于最⼤化训练数据集所有标签类别的联合预测概率
import auxlib as aul
from mxnet.gluon import data as gdata
from mxnet import autograd, nd
import sys
import time
from IPython import display
from matplotlib import pyplot as plt

def use_svg_display():
    # ⽤⽮量图显⽰。
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺⼨。
    plt.rcParams['figure.figsize'] = figsize
# 将数值标签转成相应的⽂本标签
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',  'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]
# ⼀⾏⾥画出多张图像和对应标签的函数
def show_fashion_mnist(images, labels):
    use_svg_display()
    # 这⾥的 _ 表⽰我们忽略（不使⽤）的变量。
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.reshape((28, 28)).asnumpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
# 矩阵 X 的⾏数是样本数，列数是输出个数。先通过 exp 函数对每个元素做指数运算，再对 exp 矩阵同⾏元素求和，最后令矩阵每⾏各元素与该⾏元素之和相除。 最终得到的矩阵每⾏元素和为 1 且⾮负 矩阵每⾏都是合法的概率分布
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(axis=1, keepdims=True)
    return X_exp / partition 
#定义模型
def net(X):
    return softmax(nd.dot(X.reshape((-1, num_inputs)), W) + b)
#定义损失函数
def cross_entropy(y_hat, y):
    return - nd.pick(y_hat, y).log()
#计算分类准确率
def accuracy(y_hat, y):
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()

mnist_train = gdata.vision.FashionMNIST(train=True)
mnist_test = gdata.vision.FashionMNIST(train=False)

feature, label = mnist_train[0]
print(feature.shape, feature.dtype) #高宽像素+通道数 通道数为1灰度图像
print(label, type(label), label.dtype)

X, y = mnist_train[0:9]
show_fashion_mnist(X, get_fashion_mnist_labels(y))
plt.show()

#获取并读取数据
batch_size = 256
train_iter, test_iter = aul.load_data_fashion_mnist(batch_size)

#初始化模型参数
num_inputs = 784 #28*28 = 784 该向量的每个元素对应图像中每个像素
num_outputs = 10 #图像有十个类别
W = nd.random.normal(scale=0.01, shape=(num_inputs, num_outputs))
b = nd.zeros(num_outputs)
W.attach_grad() #对模型参数附上梯度
b.attach_grad()

#训练模型
num_epochs, lr = 5, 0.1
aul.train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)

#预测
for X, y in test_iter:
    break
true_labels = get_fashion_mnist_labels(y.asnumpy())
pred_labels = get_fashion_mnist_labels(net(X).argmax(axis=1).asnumpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
show_fashion_mnist(X[0:9], titles[0:9])
plt.show()