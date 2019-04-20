from mxnet import autograd, nd
from mxnet.gluon import data as gdata
from mxnet.gluon import nn
from mxnet import init
from mxnet.gluon import loss as gloss
from mxnet import gluon

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)

batch_size = 10
dataset = gdata.ArrayDataset(features, labels)
#随机读取小批量
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)

net = nn.Sequential()
net.add(nn.Dense(1)) #Dense定义该层输出个数为 1。全连接：Dense

#初始化模型参数
net.initialize(init.Normal(sigma=0.01))  #指定权重参数每个元素将在初始化时随机采样于均值为 0 标准差为 0.01 的正态分布。

#定义损失函数
loss = gloss.L2Loss() # 平⽅损失⼜称 L2 范数损失。

#定义优化算法 学习率的数值一般设置为1/batch_size
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})  #指定学习率为 0.03 的小批量随机梯度下降（sgd）为优化算法 些参数可以通过 collect_params 函数获取

#训练模型
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size) #迭代模型参数  指明批量⼤小，从而对批量中样本梯度求平均
    l = loss(net(features), labels)
    print('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))

#获取模型参数 与构造数据集的权重偏差作对比
dense = net[0]
print(true_w, dense.weight.data())

print(true_b, dense.bias.data())
#查看梯度
print(dense.weight.grad())