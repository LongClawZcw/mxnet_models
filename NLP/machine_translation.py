import collections
import io
import math
from mxnet import autograd, gluon, init, nd
from mxnet.contrib import text
from mxnet.gluon import data as gdata, loss as gloss, nn, rnn

PAD, BOS, EOS = '<pad>', '<bos>', '<eos>'

#定义两个辅助函数对后⾯读取的数据进⾏预处理。
# 对⼀个序列，记录所有的词在 all_tokens 中以便之后构造词典，然后将该序列后添加 PAD 直到
# ⻓度变为 max_seq_len，并记录在 all_seqs 中。
def process_one_seq(seq_tokens, all_tokens, all_seqs, max_seq_len):
    all_tokens.extend(seq_tokens)
    seq_tokens += [EOS] + [PAD] * (max_seq_len - len(seq_tokens) - 1)
    all_seqs.append(seq_tokens)
# 使⽤所有的词来构造词典。并将所有序列中的词变换为词索引后构造 NDArray 实例。
def build_data(all_tokens, all_seqs):
    vocab = text.vocab.Vocabulary(collections.Counter(all_tokens), reserved_tokens=[PAD, BOS, EOS])
    indices = [vocab.to_indices(seq) for seq in all_seqs]
    return vocab, nd.array(indices)

#数据集：⼀个很小的法语—英语数据集。这个数据集⾥，每⼀⾏是⼀对法语句⼦和它对应的英语句⼦，中间使⽤'\t' 隔开。
def read_data(max_seq_len):
    # in 和 out 分别是 input 和 output 的缩写。
    in_tokens, out_tokens, in_seqs, out_seqs = [], [], [], []
    with io.open('./data/datas111.txt') as f:
        lines = f.readlines()
    for line in lines:
        in_seq, out_seq = line.rstrip().split('\t')
        in_seq_tokens, out_seq_tokens = in_seq.split(' '), out_seq.split(' ')
        if max(len(in_seq_tokens), len(out_seq_tokens)) > max_seq_len - 1:
            continue # 如果加上 EOS 后⻓于 max_seq_len，则忽略掉此样本。
        process_one_seq(in_seq_tokens, in_tokens, in_seqs, max_seq_len)
        process_one_seq(out_seq_tokens, out_tokens, out_seqs, max_seq_len)
    in_vocab, in_data = build_data(in_tokens, in_seqs)
    out_vocab, out_data = build_data(out_tokens, out_seqs)
    return in_vocab, out_vocab, gdata.ArrayDataset(in_data, out_data)

max_seq_len = 7
in_vocab, out_vocab, dataset = read_data(max_seq_len)
print(dataset[0])

#编码器
class Encoder(nn.Block):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, drop_prob=0, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=drop_prob)

    def forward(self, inputs, state):
    # 输⼊形状是（批量⼤⼩，时间步数）。将输出互换样本维和时间步维。
        embedding = self.embedding(inputs).swapaxes(0, 1)
        return self.rnn(embedding, state)
    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)

#含注意⼒机制的解码器
class Decoder(nn.Block):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, attention_size, drop_prob=0, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = attention_model(attention_size)
        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=drop_prob)
        self.out = nn.Dense(vocab_size, flatten=False)
    def forward(self, cur_input, state, enc_states):
        # 使⽤注意⼒机制计算背景向量。
        #在生成每个单词yi的时候，原先都是相同的中间语义表示C会被替换成根据当前生成单词而不断变化的Ci
        c = attention_forward(self.attention, enc_states, state[0][-1])
        # 将嵌⼊后的输⼊和背景向量在特征维连结。
        input_and_c = nd.concat(self.embedding(cur_input), c, dim=1)
        # 为输⼊和背景向量的连结增加时间步维，时间步个数为 1。
        output, state = self.rnn(input_and_c.expand_dims(0), state)
        # 移除时间步维，输出形状为（批量⼤⼩，输出词典⼤⼩）。
        output = self.out(output).squeeze(axis=0)
        return output, state
    def begin_state(self, enc_state):
        # 直接将编码器最终时间步的隐藏状态作为解码器的初始隐藏状态。
        return enc_state
#来创建⼀个批量⼤小为 4，时间步数为 7 的小批量序列输⼊ ⻔控循环单元的隐藏层个数为 2，隐藏单元个数为 16
#编码器对该输⼊执⾏前向计算后返回的输出形状为（时间步数,批量⼤小，隐藏单元个数),⻔控循环单元在最终时间步的多层隐藏状态的形状为(隐藏层个数，批量⼤小，隐藏单元个数)
encoder = Encoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
encoder.initialize()
output, state = encoder(nd.zeros((4, 7)), encoder.begin_state(batch_size=4))

#注意⼒模型的输⼊包括查询项、键项和值项,查询项为解码器在上⼀时间步的隐藏状态，形状为（批量⼤小，隐藏单元个数）,键项和值项均为编码器在所有时间步的隐藏状态，形状为（时间步数，批量⼤小，隐藏单元个数）
#注意⼒模型返回当前时间步的背景变量，形状为（批量⼤小，隐藏单元个数）
def attention_model(attention_size):
    model = nn.Sequential()
    model.add(nn.Dense(attention_size, activation='tanh', use_bias=False, flatten=False), nn.Dense(1, use_bias=False, flatten=False))
    return model

def attention_forward(model, enc_states, dec_state):
    # 将解码器隐藏状态⼴播到跟编码器隐藏状态形状相同后进⾏连结。
    dec_states = nd.broadcast_axis(dec_state.expand_dims(0), axis=0, size=enc_states.shape[0])
    enc_and_dec_states = nd.concat(enc_states, dec_states, dim=2)
    e = model(enc_and_dec_states) # 形状为（时间步数，批量⼤⼩，1）。
    alpha = nd.softmax(e, axis=0) # 在时间步维度做 softmax 运算。
    return (alpha * enc_states).sum(axis=0) # 返回背景变量。

#训练
# 实现 batch_loss 函数计算⼀个小批量的损失
def batch_loss(encoder, decoder, X, Y, loss):
    batch_size = X.shape[0]
    enc_state = encoder.begin_state(batch_size=batch_size)
    enc_outputs, enc_state = encoder(X, enc_state)
    # 初始化解码器的隐藏状态。
    dec_state = decoder.begin_state(enc_state)
    # 解码器在最初时间步的输⼊是 BOS。
    dec_input = nd.array([out_vocab.token_to_idx[BOS]] * batch_size)
    # 我们将使⽤掩码变量 mask 来忽略掉标签为填充项 PAD 的损失。
    mask, num_not_pad_tokens = nd.ones(shape=(batch_size,)), 0
    l = nd.array([0])
    for y in Y.T:
        dec_output, dec_state = decoder(dec_input, dec_state, enc_outputs)
        l = l + (mask * loss(dec_output, y)).sum()
        dec_input = y # 使⽤强制教学。
        num_not_pad_tokens += mask.sum().asscalar()
        # 当遇到 EOS 时，序列后⾯的词将均为 PAD，相应位置的掩码设成 0。
        mask = mask * (y != out_vocab.token_to_idx[EOS])
    return l / num_not_pad_tokens
#实现训练函数
def train(encoder, decoder, dataset, lr, batch_size, num_epochs):
    encoder.initialize(init.Xavier(), force_reinit=True)
    decoder.initialize(init.Xavier(), force_reinit=True)
    enc_trainer = gluon.Trainer(encoder.collect_params(), 'adam', {'learning_rate': lr})
    dec_trainer = gluon.Trainer(decoder.collect_params(), 'adam', {'learning_rate': lr})
    loss = gloss.SoftmaxCrossEntropyLoss()
    data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)
    for epoch in range(num_epochs):
        l_sum = 0.0
        for X, Y in data_iter:
            with autograd.record():
                l = batch_loss(encoder, decoder, X, Y, loss)
            l.backward()
            enc_trainer.step(1)
            dec_trainer.step(1)
            l_sum += l.asscalar()
        if (epoch + 1) % 10 == 0:
            print("epoch %d, loss %.3f" % (epoch + 1, l_sum / len(data_iter)))

#创建模型实例并设置超参数
embed_size, num_hiddens, num_layers = 64, 64, 2
attention_size, drop_prob, lr, batch_size, num_epochs = 10, 0.5, 0.01, 2, 50
encoder = Encoder(len(in_vocab), embed_size, num_hiddens, num_layers, drop_prob)
decoder = Decoder(len(out_vocab), embed_size, num_hiddens, num_layers, attention_size, drop_prob)
train(encoder, decoder, dataset, lr, batch_size, num_epochs)

# 预测 来⽣成解码器在每个时间步的输出  实现最简单的贪婪搜索
def translate(encoder, decoder, input_seq, max_seq_len):
    in_tokens = input_seq.split(' ')
    in_tokens += [EOS] + [PAD] * (max_seq_len - len(in_tokens) - 1)
    enc_input = nd.array([in_vocab.to_indices(in_tokens)])
    enc_state = encoder.begin_state(batch_size=1)
    enc_output, enc_state = encoder(enc_input, enc_state)
    dec_input = nd.array([out_vocab.token_to_idx[BOS]])
    dec_state = decoder.begin_state(enc_state)
    output_tokens = []
    for _ in range(max_seq_len):
        dec_output, dec_state = decoder(dec_input, dec_state, enc_output)
        pred = dec_output.argmax(axis=1)
        pred_token = out_vocab.idx_to_token[int(pred.asscalar())]
        if pred_token == EOS: # 当任⼀时间步搜索出 EOS 符号时，输出序列即完成。
            break
        else:
            output_tokens.append(pred_token)
            dec_input = pred
    return output_tokens


#评价机器翻译结果通常使⽤ BLEU
def bleu(pred_tokens, label_tokens, k):
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches = 0
        for i in range(len_pred - n + 1):
            if ' '.join(pred_tokens[i: i + n]) in ' '.join(label_tokens):
                num_matches += 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score
#打印函数
def score(input_seq, label_seq, k):
    pred_tokens = translate(encoder, decoder, input_seq, max_seq_len)
    label_tokens = label_seq.split(' ')
    print('bleu %.3f, predict: %s' % (bleu(pred_tokens, label_tokens, k), ' '.join(pred_tokens)))

# 简单测试⼀下模型
# input_seq = 'il a environ mon age  .'
# print(translate(encoder, decoder, input_seq, max_seq_len))

input_seq = 'No. No. No, there - There is no catch.'
print(translate(encoder, decoder, input_seq, max_seq_len))

score('I blame you, Tommy. I want to get past this.', '我要责怪你，汤米，我要卸下这个压力。', k=2)
# score('ils sont canadiens .', 'they are canadian .', k=2)

score(input_seq, translate(encoder, decoder, input_seq, max_seq_len), 2)