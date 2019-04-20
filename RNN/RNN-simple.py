import d2lzh as d2l
import load_data as ld
import math
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn, rnn
import time

#加载数据
(corpus_indices, char_to_idx, idx_to_char,
vocab_size) = ld.load_data_jay_lyrics()

#定义模型  单隐藏层、隐藏单元个数为 256
num_hiddens = 256
rnn_layer = rnn.RNN(num_hiddens)
rnn_layer.initialize()

#调⽤ rnn_layer 的成员函数 begin_state 来返回初始化的隐藏状态列表。它有⼀个形状为（隐藏层个数，批量⼤小，隐藏单元个数）的元素
batch_size = 2
state = rnn_layer.begin_state(batch_size= batch_size)
state[0].shape

num_steps = 35
X = nd.random.uniform(shape=(num_steps, batch_size, vocab_size))
Y, state_new = rnn_layer(X, state)
Y.shape, len(state_new), state_new[0].shape

#模型训练
ctx = d2l.try_gpu()
model = d2l.RNNModel(rnn_layer,vocab_size)
model.initialize(force_reinit=True, ctx=ctx)
print(d2l.predict_rnn_gluon('在一起',10, model, vocab_size, ctx, idx_to_char, char_to_idx))

#设置超参数训练模型
num_epochs, batch_size, lr, clipping_theta = 200, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
d2l.train_and_predict_rnn_gluon(model, num_hiddens, vocab_size, ctx,
                            corpus_indices, idx_to_char, char_to_idx,
                            num_epochs, num_steps, lr, clipping_theta,
                            batch_size, pred_period, pred_len, prefixes)