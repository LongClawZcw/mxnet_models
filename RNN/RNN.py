from mxnet import nd
import random
import load_data as ld
import d2lzh as d2l
import math
from mxnet import autograd, nd
from mxnet.gluon import loss as gloss
import time

(corpus_indices, char_to_idx, idx_to_char, vocab_size) = ld.load_data_jay_lyrics()
# 转换为one-hot向量实例
X = nd.arange(10).reshape((2, 5))
# inputs = ld.to_onehot(X, vocab_size)
# len(inputs), inputs[0].shape

# 初始化模型参数
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
ctx = d2l.try_gpu()
print('will use', ctx)

def get_params():
    def _one(shape):
        return nd.random.normal(scale=0.01, shape=shape, ctx=ctx)

    # 隐藏层参数。
    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = nd.zeros(num_hiddens, ctx=ctx)
    # 输出层参数。
    W_hq = _one((num_hiddens, num_outputs))
    b_q = nd.zeros(num_outputs, ctx=ctx)
    # 附上梯度。
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.attach_grad()
    return params

#定义模型
#返回初始化的隐藏状态。它返回由⼀个形状为（批量⼤小，隐藏单元个数）的值为 0 的 NDArray 组成的元组
def init_rnn_state(batch_size, num_hiddens, ctx):
    return (nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx), )
#定义rnn函数定义在一个时间步里如何计算隐藏状态和输出，激活函数使用tanh汉化（当元素在实数域上均匀分布时，tanh 函数值的均值为 0）
def rnn(inputs, state, params):
    # inputs 和 outputs 皆为 num_steps 个形状为（batch_size，vocab_size）的矩阵。
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = nd.tanh(nd.dot(X, W_xh) + nd.dot(H, W_hh) + b_h)
        Y = nd.dot(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)
#测试观察输出结果的个数（时间步数），以及第一个时间步的输出层输出形状和隐藏状态形状
state = init_rnn_state(X.shape[0], num_hiddens, ctx)
inputs = ld.to_onehot(X.as_in_context(ctx), vocab_size)
params = get_params()
outputs, state_new = rnn(inputs, state, params)
print(len(outputs), outputs[0].shape, state_new[0].shape)

#定义使用预测函数，测试predict_rnn函数，根据前缀“分开”创作⻓度为 10 个字符（不考虑前缀⻓度）的⼀段歌词。
res = d2l.predict_rnn('分开', 10, rnn, params, init_rnn_state, num_hiddens, vocab_size,ctx, idx_to_char, char_to_idx)
print(res)

#裁剪梯度在d2lzh.py

#定义模型训练函数  在d2lzh.py中

#训练模型并创作歌词
#设置超参数
num_epochs, num_steps, batch_size, lr, clipping_theta = 200, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']

#随机采样训练模型并创作歌词
d2l.train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                        vocab_size, ctx, corpus_indices, idx_to_char,
                        char_to_idx, True, num_epochs, num_steps, lr,
                        clipping_theta, batch_size, pred_period, pred_len,
                        prefixes)

#相邻采样训练模型并创作歌词
d2l.train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                        vocab_size, ctx, corpus_indices, idx_to_char,
                        char_to_idx, False, num_epochs, num_steps, lr,
                        clipping_theta, batch_size, pred_period, pred_len,
                        prefixes)
