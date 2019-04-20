import d2lzh as d2l
from mxnet import gluon, init, nd
from mxnet.gluon import nn
#inception块Inception 块⾥有四条并⾏的线路。前三条线路使⽤窗口⼤小分别是 1 × 1、3 × 3 和 5 × 5 的卷积层来抽取不同空间尺⼨下的信息。其中中间两个线路会对输⼊先做 1 × 1 卷
#积来减少输⼊通道数，以降低模型复杂度。第四条线路则使⽤ 3 × 3 最⼤池化层，后接 1 × 1 卷积层来改变通道数。四条线路都使⽤了合适的填充来使得输⼊输出⾼和宽⼀致。最后我们将每条线
#路的输出在通道维上连结，并输⼊到接下来的层中去。
class Inception(nn.Block):
    # c1 - c4 为每条线路⾥的层的输出通道数。
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 线路 1，单 1 x 1 卷积层。
        self.p1_1 = nn.Conv2D(c1, kernel_size=1, activation='relu')
        # 线路 2，1 x 1 卷积层后接 3 x 3 卷积层。
        self.p2_1 = nn.Conv2D(c2[0], kernel_size=1, activation='relu')
        self.p2_2 = nn.Conv2D(c2[1], kernel_size=3, padding=1,activation='relu')
        # 线路 3，1 x 1 卷积层后接 5 x 5 卷积层。
        self.p3_1 = nn.Conv2D(c3[0], kernel_size=1, activation='relu')
        self.p3_2 = nn.Conv2D(c3[1], kernel_size=5, padding=2,activation='relu')
        # 线路 4，3 x 3 最⼤池化层后接 1 x 1 卷积层。
        self.p4_1 = nn.MaxPool2D(pool_size=3, strides=1, padding=1)
        self.p4_2 = nn.Conv2D(c4, kernel_size=1, activation='relu')
    
    def forward(self, x):
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))
        return nd.concat(p1, p2, p3, p4, dim=1) # 在通道维上连结输出。

#主体卷积使用5个模块，每个模块之间使⽤步幅为2 的 3 × 3 最⼤池化层来减小输出⾼宽。第⼀模块使⽤⼀个 64 通道的 7 × 7 卷积层
b1 = nn.Sequential()
b1.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3, activation='relu'),nn.MaxPool2D(pool_size=3, strides=2, padding=1))
#第⼆模块使⽤两个卷积层：⾸先是 64 通道的 1 × 1 卷积层，然后是将通道增⼤ 3 倍的 3 × 3 卷积层。它对应 Inception 块中的第⼆条线路。
b2 = nn.Sequential()
b2.add(nn.Conv2D(64, kernel_size=1),nn.Conv2D(192, kernel_size=3, padding=1),nn.MaxPool2D(pool_size=3, strides=2, padding=1))
#第三模块串联两个完整的 Inception 块。
b3 = nn.Sequential()
b3.add(Inception(64, (96, 128), (16, 32), 32),Inception(128, (128, 192), (32, 96), 64),nn.MaxPool2D(pool_size=3, strides=2, padding=1))
#第四模块更加复杂。它串联了五个 Inception 块，
b4 = nn.Sequential()
b4.add(Inception(192, (96, 208), (16, 48), 64),Inception(160, (112, 224), (24, 64), 64),Inception(128, (128, 256), (24, 64), 64),Inception(112, (144, 288), (32, 64), 64),
        Inception(256, (160, 320), (32, 128), 128),nn.MaxPool2D(pool_size=3, strides=2, padding=1))

# 第五模块有输出通道数为 256 + 320 + 128 + 128 = 832 和 384 + 384 + 128 + 128 = 1024 的两个 Inception 块。其中每条线路的通道数分配思路和第三、第四模块中的⼀致，只是在具体数值上有所不同。
# 需要注意的是，第五模块的后⾯紧跟输出层，该模块同 NiN ⼀样使⽤全局平均池化层来将每个通道的⾼和宽变成 1。最后我们将输出变成⼆维数组后接上⼀个输出个数为标签类数的全连接层
b5 = nn.Sequential()
b5.add(Inception(256, (160, 320), (32, 128), 128),Inception(384, (192, 384), (48, 128), 128),nn.GlobalAvgPool2D())

net = nn.Sequential()
net.add(b1, b2, b3, b4, b5, nn.Dense(10))

X = nd.random.uniform(shape=(1, 1, 96, 96))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)


#获取数据并训练
lr, num_epochs, batch_size, ctx = 0.1, 5, 128, d2l.try_gpu()
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train_iter, test_iter = d2l.load_data_fashion_mnist1(batch_size, resize=96)
d2l.train_cnn(net, train_iter, test_iter, batch_size, trainer, ctx,num_epochs)
