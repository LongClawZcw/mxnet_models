import d2lzh as d2l
from mxnet import gluon, init, nd
from mxnet.gluon import nn

#DenseNet模型
net = nn.Sequential()
net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
        nn.BatchNorm(), nn.Activation('relu'),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1))

num_channels, growth_rate = 64, 32 # num_channels：当前的通道数。
num_convs_in_dense_blocks = [4, 4, 4, 4]
for i, num_convs in enumerate(num_convs_in_dense_blocks):
        net.add(d2l.DenseBlock(num_convs, growth_rate))
        # 上⼀个稠密的输出通道数。
        num_channels += num_convs * growth_rate
        # 在稠密块之间加⼊通道数减半的过渡层。
        if i != len(num_convs_in_dense_blocks) - 1:
                net.add(d2l.transition_block(num_channels // 2))

net.add(nn.BatchNorm(), nn.Activation('relu'), nn.GlobalAvgPool2D(), nn.Dense(10))

#获取数据并训练
lr, num_epochs, batch_size, ctx = 0.1, 5, 256, d2l.try_gpu()
net.initialize(ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train_iter, test_iter = d2l.load_data_fashion_mnist1(batch_size, resize=96)
d2l.train_cnn(net, train_iter, test_iter, batch_size, trainer, ctx,num_epochs)
