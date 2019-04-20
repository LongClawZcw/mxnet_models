#串联多个由卷积层和“全连接”层构成的小⽹络来构建⼀个深层⽹络
#以通过重复使⽤简单的基础块来构建深度模型的思路。：连续使⽤数个相同的填充为 1、窗口形状为 3 × 3 的卷积层后接上⼀个步幅为 2、窗口形状为 2 × 2 的最⼤池化层。卷积层保持输⼊的⾼和宽不变，而池化层则对其减半。
#使⽤了 8 个卷积层和 3 个全连接层，所以经常被称为 VGG-11。
import d2lzh as d2l
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn
import time

def nin_block(num_channels, kernel_size, strides, padding):
        blk = nn.Sequential()
        blk.add(nn.Conv2D(num_channels, kernel_size,
                strides, padding, activation='relu'),
                nn.Conv2D(num_channels, kernel_size=1, activation='relu'),
                nn.Conv2D(num_channels, kernel_size=1, activation='relu'))
        return blk

net = nn.Sequential()
net.add(nin_block(96, kernel_size=11, strides=4, padding=0),
        nn.MaxPool2D(pool_size=3, strides=2),
        nin_block(256, kernel_size=5, strides=1, padding=2),
        nn.MaxPool2D(pool_size=3, strides=2),
        nin_block(384, kernel_size=3, strides=1, padding=1),
        nn.MaxPool2D(pool_size=3, strides=2), nn.Dropout(0.5),
        # 标签类别数是 10。
        nin_block(10, kernel_size=3, strides=1, padding=1),
        # 全局平均池化层将窗⼝形状⾃动设置成输⼊的⾼和宽。
        nn.GlobalAvgPool2D(),
        # 将四维的输出转成⼆维的输出，其形状为（批量⼤⼩，10）。
        nn.Flatten())

X = nd.random.uniform(shape=(1, 1, 224, 224))
net.initialize()
for layer in net:
        X = layer(X)
        print(layer.name, 'output shape:\t', X.shape)
#获取数据和训练
lr, num_epochs, batch_size, ctx = 0.1, 5, 128, d2l.try_gpu()
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train_iter, test_iter = d2l.load_data_fashion_mnist1(batch_size, resize=224)
d2l.train_cnn(net, train_iter, test_iter, batch_size, trainer, ctx,num_epochs)