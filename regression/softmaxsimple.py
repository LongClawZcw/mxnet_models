import auxlib as aul
from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn
from matplotlib import pyplot as plt

#获取和读取数据
batch_size = 256
train_iter, test_iter = aul.load_data_fashion_mnist(batch_size)
#定义和初始化模型
net = nn.Sequential()
net.add(nn.Dense(10))  #添加⼀个输出个数为 10 的全连接层
net.initialize(init.Normal(sigma=0.01))
#Softmax 和交叉熵损失函数
loss = gloss.SoftmaxCrossEntropyLoss()
# 定义优化算法
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
num_epochs = 5
aul.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, trainer)
#预测
for X, y in test_iter:
    break
true_labels = aul.get_fashion_mnist_labels(y.asnumpy())
pred_labels = aul.get_fashion_mnist_labels(net(X).argmax(axis=1).asnumpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
aul.show_fashion_mnist(X[0:9], titles[0:9])
plt.show()