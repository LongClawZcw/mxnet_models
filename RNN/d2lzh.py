from mxnet import nd
import load_data as ld
import mxnet as mx
import math
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss , nn , rnn
import time
import random

def try_gpu(): 
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1,), ctx=ctx)
    except mx.base.MXNetError:
        ctx = mx.cpu()
    return ctx
def sgd(params, lr, batch_size): 
    for param in params:
        param[:] = param - lr * param.grad / batch_size

def data_iter_random(corpus_indices, batch_size, num_steps, ctx=None):
    # 减⼀是因为输出的索引是相应输⼊的索引加⼀。
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)
    # 返回从 pos 开始的⻓为 num_steps 的序列。
    def _data(pos):
        return corpus_indices[pos: pos + num_steps]
    for i in range(epoch_size):
        # 每次读取 batch_size 个随机样本。
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1) for j in batch_indices]
        yield nd.array(X, ctx), nd.array(Y, ctx)  

def data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx=None):
    corpus_indices = nd.array(corpus_indices, ctx=ctx)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0: batch_size*batch_len].reshape((
        batch_size, batch_len))
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y  
#基于前缀 prefix（含有数个字符的字符串）来预测接下来的 num_chars 个字符。
def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state,
            num_hiddens, vocab_size, ctx, idx_to_char, char_to_idx):
    state = init_rnn_state(1, num_hiddens, ctx)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        # 将上⼀时间步的输出作为当前时间步的输⼊。
        X = ld.to_onehot(nd.array([output[-1]], ctx=ctx), vocab_size)
        # 计算输出和更新隐藏状态。
        (Y, state) = rnn(X, state, params)
        # 下⼀个时间步的输⼊是 prefix ⾥的字符或者当前的最佳预测字符。
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y[0].argmax(axis=1).asscalar()))
    return ''.join([idx_to_char[i] for i in output])
#裁剪梯度
def grad_clipping(params, theta, ctx):
    norm = nd.array([0], ctx)
    for param in params:
        norm += (param.grad ** 2).sum()
    norm = norm.sqrt().asscalar()
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

#定义模型训练函数
def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                            vocab_size, ctx, corpus_indices, idx_to_char,
                            char_to_idx, is_random_iter, num_epochs, num_steps,
                            lr, clipping_theta, batch_size, pred_period,
                            pred_len, prefixes):
    if is_random_iter:
        data_iter_fn = data_iter_random
    else:
        data_iter_fn = data_iter_consecutive
    params = get_params()
    loss = gloss.SoftmaxCrossEntropyLoss()

    for epoch in range(num_epochs):
        if not is_random_iter: # 如使⽤相邻采样，在 epoch 开始时初始化隐藏状态。
            state = init_rnn_state(batch_size, num_hiddens, ctx)
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, ctx)
        for X, Y in data_iter:
            if is_random_iter: # 如使⽤随机采样，在每个⼩批量更新前初始化隐藏状态。
                state = init_rnn_state(batch_size, num_hiddens, ctx)
            else: # 否则需要使⽤ detach 函数从计算图分离隐藏状态。
                for s in state:
                    s.detach()
            with autograd.record():
                inputs = ld.to_onehot(X, vocab_size)
                # outputs 有 num_steps 个形状为（batch_size，vocab_size）的矩阵。
                (outputs, state) = rnn(inputs, state, params)
                # 拼接之后形状为（num_steps * batch_size，vocab_size）。
                outputs = nd.concat(*outputs, dim=0)
                # Y 的形状是（batch_size，num_steps），转置后再变成⻓度为
                # batch * num_steps 的向量，这样跟输出的⾏⼀⼀对应。
                y = Y.T.reshape((-1,))
                # 使⽤交叉熵损失计算平均分类误差。
                l = loss(outputs, y).mean()
            l.backward()
            grad_clipping(params, clipping_theta, ctx) # 裁剪梯度。
            sgd(params, lr, 1) # 因为误差已经取过均值，梯度不⽤再做平均。
            l_sum += l.asscalar() * y.size
            n += y.size
        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn(
                    prefix, pred_len, rnn, params, init_rnn_state,
                    num_hiddens, vocab_size, ctx, idx_to_char, char_to_idx)) 

#继承 Block 类来定义⼀个完整的循环神经⽹络。它⾸先将输⼊数据使⽤ one-hot 向量表⽰后输⼊到 rnn_layer 中，然后使⽤全连接输出层得到输出。输出个数等于词典⼤小 vocab_size。
class RNNModel(nn.Block):
    def __init__(self, rnn_layer,vocab_size , **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.dense = nn.Dense(vocab_size)
    def forward(self, inputs, state):
        # 将输⼊转置成（num_steps，batch_size）后获取 one-hot 向量表⽰。
        X = nd.one_hot(inputs.T, self.vocab_size)
        Y, state = self.rnn(X, state)
        # 全连接层会⾸先将 Y 的形状变成（num_steps * batch_size，num_hiddens），
        # 它的输出形状为（num_steps * batch_size，vocab_size）。
        output = self.dense(Y.reshape((-1, Y.shape[-1])))
        return output, state
    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)

#gluon 实现rnn的预测函数
def predict_rnn_gluon(prefix, num_chars, model, vocab_size, ctx, idx_to_char,
                        char_to_idx):
    # 使⽤ model 的成员函数来初始化隐藏状态。
    state = model.begin_state(batch_size=1, ctx=ctx)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        X = nd.array([output[-1]], ctx=ctx).reshape((1, 1))
        (Y, state) = model(X, state) # 前向计算不需要传⼊模型参数。
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y.argmax(axis=1).asscalar()))
    return ''.join([idx_to_char[i] for i in output])
#gluon 实现rnn的训练函数
def train_and_predict_rnn_gluon(model, num_hiddens, vocab_size, ctx,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes):
    loss = gloss.SoftmaxCrossEntropyLoss()
    model.initialize(ctx=ctx, force_reinit=True, init=init.Normal(0.01))
    trainer = gluon.Trainer(model.collect_params(), 'sgd',
                                {'learning_rate': lr, 'momentum': 0, 'wd': 0})
    for epoch in range(num_epochs):
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_consecutive(
            corpus_indices, batch_size, num_steps, ctx)
        state = model.begin_state(batch_size=batch_size, ctx=ctx)
        for X, Y in data_iter:
            for s in state:
                s.detach()
            with autograd.record():
                (output, state) = model(X, state)
                y = Y.T.reshape((-1,))
                l = loss(output, y).mean()
            l.backward()
            # 梯度裁剪。
            params = [p.data() for p in model.collect_params().values()]
            grad_clipping(params, clipping_theta, ctx)
            trainer.step(1) # 因为已经误差取过均值，梯度不⽤再做平均。
            l_sum += l.asscalar() * y.size
            n += y.size
        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                    epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn_gluon(
                    prefix, pred_len, model, vocab_size, ctx, idx_to_char,
                    char_to_idx))
