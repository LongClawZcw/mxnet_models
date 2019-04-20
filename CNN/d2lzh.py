# encoding: utf-8
import mxnet as mx
import math
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss , nn 
from mxnet.gluon import data as gdata
from IPython import display
from matplotlib import pyplot as plt
import sys
import time
import random
import os

#用矢量图显示
def use_svg_display():
    # ⽤⽮量图显⽰。
    display.set_matplotlib_formats('svg')
def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    #设置图的尺⼨。
    plt.rcParams['figure.figsize'] = figsize

def try_gpu(): 
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1,), ctx=ctx)
    except mx.base.MXNetError:
        ctx = mx.cpu()
    return ctx
#AlexNet 读取数据
def load_data_fashion_mnist1(batch_size, resize=None, root=os.path.join('~', '.mxnet', 'datasets', 'fashion-mnist')):
    root = os.path.expanduser(root) # 展开⽤⼾路径 '~'。
    transformer = []
    if resize:
        transformer += [gdata.vision.transforms.Resize(resize)]
    transformer += [gdata.vision.transforms.ToTensor()]
    transformer = gdata.vision.transforms.Compose(transformer)
    mnist_train = gdata.vision.FashionMNIST(root=root, train=True)
    mnist_test = gdata.vision.FashionMNIST(root=root, train=False)
    num_workers = 0 if sys.platform.startswith('win32') else 4
    train_iter = gdata.DataLoader(mnist_train.transform_first(transformer), batch_size, shuffle=True,num_workers=num_workers)
    test_iter = gdata.DataLoader(mnist_test.transform_first(transformer), batch_size, shuffle=False,num_workers=num_workers)

    return train_iter, test_iter
#load_data_fashion_mnist， 获取并读取Fashion-MNIST数据集
def load_data_fashion_mnist(batch_size):
    mnist_train = gdata.vision.FashionMNIST(train=True)
    mnist_test = gdata.vision.FashionMNIST(train=False)
    # X, y = mnist_train[0:9]
    # show_fashion_mnist(X, get_fashion_mnist_labels(y))
    #batch_size = 256
    transformer = gdata.vision.transforms.ToTensor()
    if sys.platform.startswith('win'):
        num_workers = 0 # 0 表⽰不⽤额外的进程来加速读取数据。
    else:
        num_workers = 4
    train_iter = gdata.DataLoader(mnist_train.transform_first(transformer),batch_size, shuffle=True,num_workers=num_workers)
    test_iter = gdata.DataLoader(mnist_test.transform_first(transformer),batch_size, shuffle=False,num_workers=num_workers)

    return train_iter,test_iter

#图像的标签使用NumPy的标量表示，以下函数将数值标签转成相应的文本标签
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat','sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

#在⼀⾏⾥画出多张图像和对应标签的函数
def show_fashion_mnist(images, labels):
    use_svg_display()
    # 这⾥的 _ 表⽰我们忽略（不使⽤）的变量。
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.reshape((28, 28)).asnumpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)

#二维互相关运算
def corr2d(X,K):
    h, w = K.shape
    Y = nd.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y

#二维卷积层   输⼊和卷积核做互相关运算，并加上⼀个标量偏差来得到输出  们先对卷积核随机初始化，然后不断迭代卷积核和偏差。
class Conv2D(nn.Block):
    def __init__(self, kernel_size, **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=kernel_size)
        self.bias = self.params.get('bias', shape=(1,))
    def forward(self, x):
        return corr2d(x, self.weight.data()) + self.bias.data()

#定义⼀个便利函数来计算卷积层。它初始化卷积层权重，并对输⼊和输出做相应的升维和降维。
def comp_conv2d(conv2d, X):
    conv2d.initialize()
    # （1，1）代表批量⼤⼩和通道数（后⾯章节将介绍）均为 1。
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:]) # 排除不关⼼的前两维：批量和通道。

#实现多输入通道的互相关运算，只需要对每一个通道做互相关运算，然后通过add_n函数进行累加
def corr2d_multi_in(X, K):
    # 我们⾸先沿着 X 和 K 的第 0 维（通道维）遍历。然后使⽤ * 将结果列表变成 add_n 函数
    # 的位置参数（positional argument）来进⾏相加。
    return nd.add_n(*[corr2d(x, k) for x, k in zip(X, K)])

#实现一个互相关运算函数计算多个通道的输出
def corr2d_multi_in_out(X,K):
    # 对 K 的第 0 维遍历，每次同输⼊ X 做互相关计算。所有结果使⽤ stack 函数合并在⼀起。
    return nd.stack(*[corr2d_multi_in(X, k) for k in K])

# 使⽤全连接层中的矩阵乘法来实现 1 × 1 卷积
# def corr2d_multi_in_out_1x1(X, K):
#     c_i, h, w = X.shape
#     c_o = K.shape[0]
#     X = X.reshape((c_i, h * w))
#     K = K.reshape((c_o, c_i))
#     Y = nd.dot(K, X) # 全连接层的矩阵乘法。
#     return Y.reshape((c_o, h, w))

#池化层的前向计算
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = nd.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y

def evaluate_accuracy(data_iter, net, ctx):
    acc_sum, n = nd.array([0], ctx=ctx), 0
    for X, y in data_iter:
        # 如果 ctx 代表 GPU 及相应的显存，将数据复制到显存上。
        X, y = X.as_in_context(ctx), y.as_in_context(ctx).astype('float32')
        acc_sum += (net(X).argmax(axis=1) == y).sum()
        n += y.size
    return acc_sum.asscalar() / n

#CNN训练函数
def train_cnn(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs):
    print('training on', ctx)
    loss = gloss.SoftmaxCrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X, y = X.as_in_context(ctx), y.as_in_context(ctx)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).sum()
            l.backward()
            trainer.step(batch_size)
            y = y.astype('float32')
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size
        test_acc = evaluate_accuracy(test_iter, net, ctx)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, '
        'time %.1f sec'
        % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc,
        time.time() - start))

#批量归一化
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 通过 autograd 来判断当前模式为训练模式或预测模式。
    if not autograd.is_training():
    # 如果是在预测模式下，直接使⽤传⼊的移动平均所得的均值和⽅差。
        X_hat = (X - moving_mean) / nd.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 使⽤全连接层的情况，计算特征维上的均值和⽅差。
            mean = X.mean(axis=0)
            var = ((X - mean) ** 2).mean(axis=0)
        else:
            # 使⽤⼆维卷积层的情况，计算通道维上（axis=1）的均值和⽅差。这⾥我们需要
            # 保持 X 的形状以便后⾯可以做⼴播运算。
            mean = X.mean(axis=(0, 2, 3), keepdims=True)
            var = ((X - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
        # 训练模式下⽤当前的均值和⽅差做标准化。
        X_hat = (X - mean) / nd.sqrt(var + eps)
        # 更新移动平均的均值和⽅差。
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta # 拉伸和偏移。
    return Y, moving_mean, moving_var

#batchNorm层，存参与求梯度和迭代的拉伸参数 gamma 和偏移参数 beta，同时也维护移动平均得到的均值和⽅差，以能够在模型预测时使⽤
#BatchNorm 实例所需指定的 num_features 参数对于全连接层为输出个数，对于卷积层则为输出通道数。
class BatchNorm(nn.Block):
    def __init__(self, num_features, num_dims, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成 0 和 1。
        self.gamma = self.params.get('gamma', shape=shape, init=init.One())
        self.beta = self.params.get('beta', shape=shape, init=init.Zero())
        # 不参与求梯度和迭代的变量，全在 CPU 上初始化成 0。
        self.moving_mean = nd.zeros(shape)
        self.moving_var = nd.zeros(shape)
    def forward(self, X):
    # 如果 X 不在 CPU 上，将 moving_mean 和 moving_var 复制到 X 所在设备上。
        if self.moving_mean.context != X.context:
            self.moving_mean = self.moving_mean.copyto(X.context)
            self.moving_var = self.moving_var.copyto(X.context)
    # 保存更新过的 moving_mean 和 moving_var。
        Y, self.moving_mean, self.moving_var = batch_norm(X, self.gamma.data(), self.beta.data(), self.moving_mean,self.moving_var, eps=1e-5, momentum=0.9)
        return Y

#残差块的实现 ⾸先有两个有同样输出通道数的 3 × 3 卷积层。每个卷积层后接⼀个批量归⼀化层和 ReLU 激活函数。然后我们将输⼊跳过这两个卷积运算后直接加在最后的 ReLU 激活函数前。
class Residual(nn.Block):
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1,strides=strides)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1,strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()
    def forward(self, X):
        Y = nd.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return nd.relu(Y + X)

#稠密块的实现
def conv_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(), nn.Activation('relu'), nn.Conv2D(num_channels, kernel_size=3, padding=1))
    return blk
class DenseBlock(nn.Block):
    def __init__(self, num_convs, num_channels, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        self.net = nn.Sequential()
        for _ in range(num_convs):
            self.net.add(conv_block(num_channels))

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = nd.concat(X, Y, dim=1) # 在通道维上将输⼊和输出连结。
        return X
#过渡层 用来控制稠密连接网络的复杂度
def transition_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(), nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=1),
            nn.AvgPool2D(pool_size=2, strides=2))
    return blk