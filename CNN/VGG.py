#使⽤重复元素的⽹络（VGG）
#以通过重复使⽤简单的基础块来构建深度模型的思路。：连续使⽤数个相同的填充为 1、窗口形状为 3 × 3 的卷积层后接上⼀个步幅为 2、窗口形状为 2 × 2 的最⼤池化层。卷积层保持输⼊的⾼和宽不变，而池化层则对其减半。
#使⽤了 8 个卷积层和 3 个全连接层，所以经常被称为 VGG-11。
import d2lzh as d2l
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn
import time

def vgg_block(num_convs, num_channels):
        blk = nn.Sequential()
        for _ in range(num_convs):
                blk.add(nn.Conv2D(num_channels, kernel_size=3,
                padding=1, activation='relu'))
        blk.add(nn.MaxPool2D(pool_size=2, strides=2))
        return blk
def vgg(conv_arch):
        net = nn.Sequential()
        # 卷积层部分。
        for (num_convs, num_channels) in conv_arch:
                net.add(vgg_block(num_convs, num_channels))
        # 全连接层部分。
        net.add(nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
                nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
                nn.Dense(10))
        return net

conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
net = vgg(conv_arch)
net.initialize()
X = nd.random.uniform(shape=(1, 1, 224, 224))
for blk in net:
        X = blk(X)
        print(blk.name, 'output shape:\t', X.shape)

# #获取数据和训练
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)
lr, num_epochs, batch_size, ctx = 0.05, 5, 128, d2l.try_gpu()
net.initialize(ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train_iter, test_iter = d2l.load_data_fashion_mnist1(batch_size, resize=224)
d2l.train_cnn(net, train_iter, test_iter, batch_size, trainer, ctx,
num_epochs)