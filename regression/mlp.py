#多层感知机在单层神经⽹络的基础上引⼊了⼀到多个隐藏层，隐藏层位于输⼊层和输出层之间。
# 常⽤的激活函数包括 ReLU 函数、sigmoid 函数和 tanh 函数。 非线性函数
from mxnet import nd
from mxnet.gluon import loss as gloss
import auxlib as aul
from matplotlib import pyplot as plt
#使⽤多层感知机对图像进⾏分类 使⽤ Fashion-MNIST 数据集
#获取和读取数据
batch_size = 256
train_iter, test_iter = aul.load_data_fashion_mnist(batch_size)

#定义激活函数relu
def relu(X):
    return nd.maximum(X, 0)

#定义模型参数
num_inputs, num_outputs, num_hiddens = 784, 10, 256  #隐藏单元个数为 256。
W1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens))
b1 = nd.zeros(num_hiddens)
W2 = nd.random.normal(scale=0.01, shape=(num_hiddens, num_outputs))
b2 = nd.zeros(num_outputs)
params = [W1, b1, W2, b2]

for param in params:
    param.attach_grad()

#定义模型
def net(X):
    X = X.reshape((-1, num_inputs)) #将每张原始图像改成⻓度为 num_inputs 的向量
    H = relu(nd.dot(X, W1) + b1)
    return nd.dot(H, W2) + b2
#定义损失函数
loss = gloss.SoftmaxCrossEntropyLoss()
#训练模型
num_epochs, lr = 5, 0.5
aul.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)

#预测
for X, y in test_iter:
    break
true_labels = aul.get_fashion_mnist_labels(y.asnumpy())
pred_labels = aul.get_fashion_mnist_labels(net(X).argmax(axis=1).asnumpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
aul.show_fashion_mnist(X[0:9], titles[0:9])
plt.show()