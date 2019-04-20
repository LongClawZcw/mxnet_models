from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn
import auxlib as aul
from matplotlib import pyplot as plt

#定义模型
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'), nn.Dense(256, activation='relu'), nn.Dense(10)) #多加了⼀个全连接层作为隐藏层。它的隐藏单元个数为256，并使⽤ ReLU 作为激活函数。
net.initialize(init.Normal(sigma=0.01))

#读取数据并训练模型
batch_size = 256
train_iter, test_iter = aul.load_data_fashion_mnist(batch_size)
#定义损失函数
loss = gloss.SoftmaxCrossEntropyLoss()
#训练
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})
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