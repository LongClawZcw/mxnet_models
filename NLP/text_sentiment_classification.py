import collections
import d2lzh as d2l
from mxnet import gluon, init, nd
from mxnet.contrib import text
from mxnet.gluon import data as gdata, loss as gloss, nn, rnn, utils as gutils
import os
import random
import tarfile

#使⽤ Stanford’s Large Movie Review Dataset 作为⽂本情感分类的数据集
d2l.download_imdb()

train_data, test_data = d2l.read_imdb('train'), d2l.read_imdb('test')

#数据预处理 对每条评论做分词，基于空格进⾏分词，从而得到分好词的评论。  过滤掉了出现次数少于 5 的词。

vocab = d2l.get_vocab_imdb(train_data)
print('分词：')
print('# words in vocab:', len(vocab))
print(train_data[0])
#创建数据迭代器。每次迭代将返回⼀个小批量的数据。
#preprocess_imdb函数对每条评论进⾏分词，并通过词典转换成词索引，然后通过截断或者补 0 来将每条评论⻓度固定成500。
batch_size = 64
train_set = gdata.ArrayDataset(*d2l.preprocess_imdb(train_data, vocab)) 
test_set = gdata.ArrayDataset(*d2l.preprocess_imdb(test_data, vocab))
train_iter = gdata.DataLoader(train_set, batch_size, shuffle=True)
test_iter = gdata.DataLoader(test_set, batch_size)

#构建循环神经网络的模型 
#每个词先通过嵌⼊层得到特征向量。然后，我们使⽤双向循环神经⽹络对特征序列进⼀步编码得到序列信息。最后，我们将编码的序列信息通过全连接层变换为输出
# 创建⼀个含两个隐藏层的双向循环神经⽹络
embed_size, num_hiddens, num_layers, ctx = 100, 100, 2, d2l.try_all_gpus()
net = d2l.BiRNN(vocab, embed_size, num_hiddens, num_layers)
net.initialize(init.Xavier(), ctx=ctx)

#加载预训练的词向量 ⽤这些词向量作为评论中每个词的特征向量
glove_embedding = text.embedding.create('glove', pretrained_file_name='glove.6B.100d.txt', vocabulary=vocab)

#预训练词向量的维度需要跟创建的模型中的嵌⼊层输出⼤小 embed_size ⼀致 在训练中我们不再更新这些词向量
net.embedding.weight.set_data(glove_embedding.idx_to_vec)
net.embedding.collect_params().setattr('grad_req', 'null')

#训练并评价模型
lr, num_epochs = 0.01, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gloss.SoftmaxCrossEntropyLoss()
d2l.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs)

#预测
print(d2l.predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'great']))