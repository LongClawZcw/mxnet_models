import d2lzh as d2l
from mxnet import gluon, init, nd
from mxnet.contrib import text
from mxnet.gluon import data as gdata, loss as gloss, nn

# TextCNN 中使⽤的时序最⼤池化层  假设输⼊包含多个通道，各通道由不同时间步上的数值组成，各通道的输出即该通道所有时间步中最⼤的数值。
# 读取和预处理 IMDb 数据集
batch_size = 64
d2l.download_imdb()
train_data, test_data = d2l.read_imdb('train'), d2l.read_imdb('test')
vocab = d2l.get_vocab_imdb(train_data)
train_iter = gdata.DataLoader(gdata.ArrayDataset(*d2l.preprocess_imdb(train_data, vocab)), batch_size, shuffle=True)
test_iter = gdata.DataLoader(gdata.ArrayDataset(*d2l.preprocess_imdb(test_data, vocab)), batch_size)

#TextCNN模型 使⽤⼀维卷积层和时序最⼤池化层

# 创建⼀个 TextCNN 实例。它有 3 个卷积层，它们的核宽分别为 3、4 和 5，输出通道数均为 100。
embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
ctx = d2l.try_all_gpus()
net = d2l.TextCNN(vocab, embed_size, kernel_sizes, nums_channels)
net.initialize(init.Xavier(), ctx=ctx)

#加载预训练的词向量 加载预训练的 100 维 GloVe 词向量，并分别初始化嵌⼊层 embedding 和 constant_embedding。其中前者参与训练，而后者权重固定。
glove_embedding = text.embedding.create('glove', pretrained_file_name='glove.6B.100d.txt', vocabulary=vocab)
net.embedding.weight.set_data(glove_embedding.idx_to_vec)
net.constant_embedding.weight.set_data(glove_embedding.idx_to_vec)
net.constant_embedding.collect_params().setattr('grad_req', 'null')

# 训练并评价模型
lr, num_epochs = 0.001, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gloss.SoftmaxCrossEntropyLoss()
d2l.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs)

#情感预测
d2l.predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'bad'])