# encoding: utf-8
#深度卷积神经网络的实现，
import d2lzh as d2l
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, nn
import sys
import os

net = nn.Sequential()
# 使⽤较⼤的 11 x 11 窗⼝来捕获物体。同时使⽤步幅 4 来较⼤减⼩输出⾼和宽。
# 这⾥使⽤的输出通道数⽐ LeNet 中的也要⼤很多。
net.add(nn.Conv2D(96, kernel_size=11, strides=4, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        # 减⼩卷积窗⼝，使⽤填充为 2 来使得输⼊输出⾼宽⼀致，且增⼤输出通道数。
        nn.Conv2D(256, kernel_size=5, padding=2, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        # 连续三个卷积层，且使⽤更⼩的卷积窗⼝。除了最后的卷积层外，进⼀步增⼤了输出通道数。
        # 前两个卷积层后不使⽤池化层来减⼩输⼊的⾼和宽。
        nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
        nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
        nn.Conv2D(256, kernel_size=3, padding=1, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        # 这⾥全连接层的输出个数⽐ LeNet 中的⼤数倍。使⽤丢弃层来缓解过拟合。
        nn.Dense(4096, activation="relu"), nn.Dropout(0.5),
        nn.Dense(4096, activation="relu"), nn.Dropout(0.5),
        # 输出层。由于这⾥使⽤ Fashion-MNIST，所以⽤类别数为 10，⽽⾮论⽂中的 1000。
        nn.Dense(10))

#构造⼀个⾼和宽均为 224 的单通道数据样本来观察每⼀层的输出形状。
X = nd.random.uniform(shape=(1, 1, 224, 224))
net.initialize()
for layer in net:
        X = layer(X)
        print(layer.name, 'output shape:\t', X.shape)


#读取数据
batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist1(batch_size, resize=224)
lr, num_epochs, ctx = 0.01, 5, d2l.try_gpu()
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
d2l.train_cnn(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)
