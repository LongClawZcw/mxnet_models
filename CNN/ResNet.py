import d2lzh as d2l
from mxnet import gluon, init, nd
from mxnet.gluon import nn

#ResNet模型
net = nn.Sequential()
net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
        nn.BatchNorm(), nn.Activation('relu'),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1))

#ResNet 使⽤四个由残差块组成的模块，每个模块使⽤若⼲个同样输出通道数的残差块。第⼀个模块的通道数同输⼊通道数⼀致。由于之前已经使⽤了步幅为 2 的最⼤池化层，
# 所以⽆需减小⾼和宽。之后的每个模块在第⼀个残差块⾥将上⼀个模块的通道数翻倍，并减半⾼和宽。
def resnet_block(num_channels, num_residuals, first_block=False):
    blk = nn.Sequential()
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.add(d2l.Residual(num_channels, use_1x1conv=True, strides=2))
        else:
            blk.add(d2l.Residual(num_channels))
    return blk

net.add(resnet_block(64, 2, first_block=True),
        resnet_block(128, 2),
        resnet_block(256, 2),
        resnet_block(512, 2))
#加⼊全局平均池化层后接上全连接层输出
net.add(nn.GlobalAvgPool2D(), nn.Dense(10))

X = nd.random.uniform(shape=(1, 1, 224, 224))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)

#获取数据并训练
lr, num_epochs, batch_size, ctx = 0.05, 5, 256, d2l.try_gpu()
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train_iter, test_iter = d2l.load_data_fashion_mnist1(batch_size, resize=96)
d2l.train_cnn(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)