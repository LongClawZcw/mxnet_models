#卷积神经网络的实现，含卷积层的网络，早期用来识别手写数字图像的卷积神经网络
# 在卷积层块中，每个卷积层都使⽤ 5×5 的窗口，并在输出上使⽤ sigmoid 激活函数。第⼀个卷积层输出通道数为 6，第⼆个卷积层输出通道数则增加到 16。这是因为第⼆个卷积层⽐第⼀个卷积层的输⼊的⾼和宽要小，所以增加输出通道使两个卷积层的参数尺⼨类似。
# 卷积层块的两个最⼤池化层的窗口形状均为 2 × 2，且步幅为 2。由于池化窗口与步幅形状相同，池化窗口在输⼊上每次滑动所覆盖的区域互不重叠。

# 卷积层块的输出形状为（批量⼤小，通道，⾼，宽）。当卷积层块的输出传⼊全连接层块时，全连接层块会将小批量中每个样本变平（flatten）。也就是说，全连接层的输⼊形状将变成⼆维，其中
# 第⼀维为小批量中的样本，第⼆维为每个样本变平后的向量表⽰，且向量⻓度为通道、⾼和宽的乘积。全连接层块含三个全连接层。它们的输出个数分别是 120、84 和 10。其中 10 为输出的类
# 别个数。
import d2lzh as d2l
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn
import time

# net = nn.Sequential()
# net.add(nn.Conv2D(channels=6, kernel_size=5,activation='sigmoid'),
#         nn.MaxPool2D(pool_size=2, strides=2),
#         nn.Conv2D(channels=16, kernel_size=5, activation='sigmoid'),
#         nn.MaxPool2D(pool_size=2, strides=2),
#         # Dense 会默认将（批量⼤⼩，通道，⾼，宽）形状的输⼊转换成
#         # （批量⼤⼩，通道 * ⾼ * 宽）形状的输⼊。
#         nn.Dense(120, activation='sigmoid'),
#         nn.Dense(84, activation='sigmoid'),
#         nn.Dense(10)
# )
#使用批量归一化层后的LeNet
net = nn.Sequential()
net.add(nn.Conv2D(6, kernel_size=5),
        # d2l.BatchNorm(6, num_dims=4),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(16, kernel_size=5),
        d2l.BatchNorm(16, num_dims=4),
        nn.Activation('sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Dense(120),
        d2l.BatchNorm(120, num_dims=2),
        nn.Activation('sigmoid'),
        nn.Dense(84),
        d2l.BatchNorm(84, num_dims=2),
        nn.Activation('sigmoid'),
        nn.Dense(10))
#训练修改后的模型
lr, num_epochs, batch_size, ctx = 1.0, 5, 256, d2l.try_gpu()
net.initialize(ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_cnn(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)

#获取数据和训练
# batch_size = 256
# train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size= batch_size)

# lr, num_epochs = 0.9, 5
# ctx = d2l.try_gpu()
# net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())

# trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
# d2l.train_cnn(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)