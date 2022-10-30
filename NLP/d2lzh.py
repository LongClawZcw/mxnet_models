
import random
import math
import mxnet as mx
from mxnet import autograd, gluon, init, nd

import collections
from mxnet import gluon, init, nd
from mxnet.contrib import text
from mxnet.gluon import data as gdata, loss as gloss, nn, rnn, utils as gutils
import os
import time
import tarfile


def try_gpu(): 
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1,), ctx=ctx)
    except mx.base.MXNetError:
        ctx = mx.cpu()
    return ctx

def try_all_gpus():
    ctxes = []
    try:
        for i in range(16): # 假设⼀台机器上 GPU 的个数不超过 16。
            ctx = mx.gpu(i)
            _ = nd.array([0], ctx=ctx)
            ctxes.append(ctx)
    except mx.base.MXNetError:
        pass
    if not ctxes:
        ctxes = [mx.cpu()]
    return ctxes

def _get_batch(batch, ctx):
    features, labels = batch
    if labels.dtype != features.dtype:
        labels = labels.astype(features.dtype)
        # 当 ctx 包含多个 GPU 时，划分⼩批量数据样本并复制到各个 GPU 上。
    return (gutils.split_and_load(features, ctx), gutils.split_and_load(labels, ctx), features.shape[0])

def evaluate_accuracy(data_iter, net, ctx=[mx.cpu()]):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc_sum, n = nd.array([0]), 0
    for batch in data_iter:
        features, labels, _ = _get_batch(batch, ctx)
        for X, y in zip(features, labels):
            y = y.astype('float32')
            acc_sum += (net(X).argmax(axis=1) == y).sum().copyto(mx.cpu())
            n += y.size
        acc_sum.wait_to_read()
    return acc_sum.asscalar() / n

#下载 Stanford’s Large Movie Review Dataset 作为⽂本情感分类的数据集
def download_imdb(data_dir='./data'):
    url = ('http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz')
    sha1 = '01ada507287d82875905620988597833ad4e0903'
    fname = gutils.download(url, data_dir, sha1_hash=sha1)
    with tarfile.open(fname, 'r') as f:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(f, data_dir)
#读取
def read_imdb(folder='train'): 
    data = []
    for label in ['pos', 'neg']:
        folder_name = os.path.join('./data/aclImdb/', folder, label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '').lower()
                data.append([review, 1 if label == 'pos' else 0])
    random.shuffle(data)
    return data

#数据预处理-分词（基于空格）
def get_tokenized_imdb(data): # 本函数已保存在 d2lzh 包中⽅便以后使⽤。
    def tokenizer(text):
        return [tok.lower() for tok in text.split(' ')]
    return [tokenizer(review) for review, _ in data]

#数据预处理-过滤掉出现次数少于 5 的词。
def get_vocab_imdb(data): 
    tokenized_data = get_tokenized_imdb(data)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    return text.vocab.Vocabulary(counter, min_freq=5)

#因为每条评论⻓度不⼀致使得不能直接组合成小批量，我们定义 preprocess_imdb 函数对每条评论进⾏分词，并通过词典转换成词索引，然后通过截断或者补 0 来将每条评论⻓度固定成
#500。
def preprocess_imdb(data, vocab): # 本函数已保存在 d2lzh 包中⽅便以后使⽤。
    max_l = 500 # 将每条评论通过截断或者补 0，使得⻓度变成 500。
    def pad(x):
        return x[:max_l] if len(x) > max_l else x + [0] * (max_l - len(x))
    tokenized_data = get_tokenized_imdb(data)
    features = nd.array([pad(vocab.to_indices(x)) for x in tokenized_data])
    labels = nd.array([score for _, score in data])
    return features, labels

#双向循环神经网络
class BiRNN(nn.Block):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        # bidirectional 设 True 即得到双向循环神经⽹络。
        self.encoder = rnn.LSTM(num_hiddens, num_layers=num_layers,
        bidirectional=True, input_size=embed_size)
        self.decoder = nn.Dense(2)
    def forward(self, inputs):
        # inputs 的形状是（批量⼤⼩，词数），因为 LSTM 需要将序列作为第⼀维，所以将输⼊转
        # 置后再提取词特征，输出形状为（词数，批量⼤⼩，词向量维度）。
        embeddings = self.embedding(inputs.T)
        # states 形状是（词数，批量⼤⼩，2 * 隐藏单元个数）。
        states = self.encoder(embeddings)
        # 连结初始时间步和最终时间步的隐藏状态作为全连接层输⼊。它的形状为（批量⼤⼩， # 4 * 隐藏单元个数）。
        encoding = nd.concat(states[0], states[-1])
        outputs = self.decoder(encoding)
        return outputs


#定义 train 函数使⽤多 GPU 训练并评价模型
def train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs):
    print('training on', ctx)
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, m, start = 0.0, 0.0, 0, 0, time.time()
        for i, batch in enumerate(train_iter):
            Xs, ys, batch_size = _get_batch(batch, ctx)
            ls = []
            with autograd.record():
                y_hats = [net(X) for X in Xs]
                ls = [loss(y_hat, y) for y_hat, y in zip(y_hats, ys)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
            train_l_sum += sum([l.sum().asscalar() for l in ls])
            n += sum([l.size for l in ls])
            train_acc_sum += sum([(y_hat.argmax(axis=1) == y).sum().asscalar() 
                                for y_hat, y in zip(y_hats, ys)])
            m += sum([y.size for y in ys])
        test_acc = evaluate_accuracy(test_iter, net, ctx)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, '
            'time %.1f sec'
            % (epoch + 1, train_l_sum / n, train_acc_sum / m, test_acc, time.time() - start))

#定义预测函数
def predict_sentiment(net, vocab, sentence):
    sentence = nd.array(vocab.to_indices(sentence), ctx=try_gpu())
    label = nd.argmax(net(sentence.reshape((1, -1))), axis=1)
    return 'positive' if label.asscalar() == 1 else 'negative'

###################################################
#text-CNN 使用卷积循环网络进行情感分析
#⼀维互相关运算
def corr1d(X, K):
    w = K.shape[0]
    Y = nd.zeros((X.shape[0] - w + 1))
    for i in range(Y.shape[0]):
        Y[i] = (X[i: i + w] * K).sum()
    return Y
# 多输⼊通道的⼀维互相关运算 主要使⽤了⼀维卷积层和时序最⼤池化层
def corr1d_multi_in(X, K):
    # 我们⾸先沿着 X 和 K 的第 0 维（通道维）遍历。然后使⽤ * 将结果列表变成 add_n 函数
    # 的位置参数（positional argument）来进⾏相加。
    return nd.add_n(*[corr1d(x, k) for x, k in zip(X, K)])

# textCNN 模型
class TextCNN(nn.Block):
    def __init__(self, vocab, embed_size, kernel_sizes, num_channels, **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        # 不参与训练的嵌⼊层。
        self.constant_embedding = nn.Embedding(len(vocab), embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Dense(2)
        # 时序最⼤池化层没有权重，所以可以共⽤⼀个实例。
        self.pool = nn.GlobalMaxPool1D()
        self.convs = nn.Sequential() # 创建多个⼀维卷积层。
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.add(nn.Conv1D(c, k, activation='relu'))
    
    def forward(self, inputs):
        # 将两个形状是（批量⼤⼩，词数，词向量维度）的嵌⼊层的输出按词向量连结。
        embeddings = nd.concat(self.embedding(inputs), self.constant_embedding(inputs), dim=2)
        # 根据 Conv1D 要求的输⼊格式，将词向量维，即⼀维卷积层的通道维，变换到前⼀维。
        embeddings = embeddings.transpose((0, 2, 1))
        # 对于每个⼀维卷积层，在时序最⼤池化后会得到⼀个形状为（批量⼤⼩，通道⼤⼩，1）的
        # NDArray。使⽤ flatten 函数去掉最后⼀维，然后在通道维上连结。
        encoding = nd.concat(*[nd.flatten(self.pool(conv(embeddings))) for conv in self.convs], dim=1)
        # 应⽤丢弃法后使⽤全连接层得到输出。
        outputs = self.decoder(self.dropout(encoding))
        return outputs
    