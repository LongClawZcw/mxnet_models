import collections
import d2lzh as d2l
import math
from mxnet import autograd, gluon, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
import random
import sys
import time
import zipfile

#处理数据集 Penn Tree Bank（PTB）是⼀个常⽤的小型语料库 它采样⾃华尔街⽇报的⽂章，包括训练集、验证集和测试集
with zipfile.ZipFile('./data/ptb.zip','r') as zin:
    zin.extractall('./data/')

with open('./data/ptb/ptb.train.txt','r') as f:
    lines = f.readlines()
    # st 是 sentence 在循环中的缩写。
    raw_dataset = [st.split() for st in lines]

#对于数据集的前三个句⼦，打印每个句⼦的词数和前五个词。这个数据集中句尾符为“<eos>”，
#⽣僻词全⽤“<unk>”表⽰，数字则被替换成了“N”。
for st in raw_dataset[:3]:
    print('# tokens:', len(st), st[:5])

#建立词语索引 们只保留在数据集中⾄少出现 5 次的词。
counter = collections.Counter([tk for st in raw_dataset for tk in st])
counter = dict(filter(lambda x: x[1] >= 5, counter.items()))
#将词映射到整数索引
idx_to_token = [tk for tk, _ in counter.items()]
token_to_idx = {tk: idx for idx, tk in enumerate(idx_to_token)}
dataset = [[token_to_idx[tk] for tk in st if tk in token_to_idx]
            for st in raw_dataset]
num_tokens = sum([len(st) for st in dataset])
print('# tokens: %d' % num_tokens)
# print(dataset[0:3])

#二次采样 且越⾼频的词被丢弃的概率越⼤
def discard(idx):
    return random.uniform(0, 1) < 1 - math.sqrt(1e-4 / counter[idx_to_token[idx]] * num_tokens)

subsampled_dataset = [[tk for tk in st if not discard(tk)] for st in dataset]
print('# tokens: %d' % sum([len(st) for st in subsampled_dataset]))

#⽐较⼀个词在⼆次采样前后出现在数据集中的次数。
def compare_counts(token):
    return '# %s: before=%d, after=%d' % (token, sum(
        [st.count(token_to_idx[token]) for st in dataset]), sum(
        [st.count(token_to_idx[token]) for st in subsampled_dataset]))

print(compare_counts('the'))
print(compare_counts('join'))

#提取中心词和背景词，与中心词距离不超过背景窗口大小的词作为背景词  每次在整数 1 和 max_window_size（最⼤背景窗口）之间均匀随机采样⼀个整数作为背景窗口⼤小
def get_centers_and_contexts(dataset, max_window_size):
    centers, contexts = [], []
    for st in dataset:
        if len(st) < 2: # 每个句⼦⾄少要有 2 个词才可能组成⼀对“中⼼词 - 背景词”。
            continue
        centers += st
        for center_i in range(len(st)):
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, center_i - window_size), min(len(st), center_i + 1 + window_size)))
            indices.remove(center_i) # 将中⼼词排除在背景词之外。
            contexts.append([st[idx] for idx in indices])
    return centers, contexts

#创建人工数据集，其中含有词数为7和3的两个句子，最大背景窗口为2，打印所有背景词和中心词
print('人工数据集测试：')
tiny_dataset = [list(range(7)), list(range(7, 10))]
print('dataset', tiny_dataset)
for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
    print('center', center, 'has contexts', context)

#将最大窗口设为4，提取数据集中所有中心词和背景词
all_centers, all_contexts = get_centers_and_contexts(subsampled_dataset, 5)

#负采样 对于⼀对中⼼词和背景词，我们随机采样 K 个噪⾳词 噪⾳词采样概率 P(w) 设为 w 词频与总词频之⽐的 0.75次⽅
def get_negatives(all_contexts, sampling_weights, K):
    all_negatives, neg_candidates, i = [], [], 0
    population = list(range(len(sampling_weights)))
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            if i == len(neg_candidates):
                # 根据每个词的权重（sampling_weights）随机⽣成 k 个词的索引作为噪⾳
                # 词。为了⾼效计算，可以将 k 设的稍⼤⼀点。
                i, neg_candidates = 0, random.choices(
                population, sampling_weights, k=int(1e5))
            neg, i = neg_candidates[i], i + 1
            # 噪⾳词不能是背景词。
            if neg not in set(contexts):
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives

sampling_weights = [counter[w]**0.75 for w in idx_to_token]
all_negatives = get_negatives(all_contexts, sampling_weights, 5)

#数据处理完毕，小批量读取数据all_centers、all_contexts、all_negatives(噪音词)
def batchify(data):
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (nd.array(centers).reshape((-1, 1)), nd.array(contexts_negatives),
            nd.array(masks), nd.array(labels))

#⽤刚刚定义的 batchify 函数指定 DataLoader 实例中小批量的读取⽅式。然后打印读取的第⼀个批量中各个变量的形状。
batch_size = 512
num_workers = 0 if sys.platform.startswith('win32') else 4
dataset = gdata.ArrayDataset(all_centers, all_contexts, all_negatives)
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True, batchify_fn=batchify, num_workers=num_workers)
for batch in data_iter:
    for name, data in zip(['centers', 'contexts_negatives', 'masks','labels'], batch):
        print(name, 'shape:', data.shape)
    break

#跳字模型 使⽤嵌⼊层和小批量乘法来实现跳字模型实现 嵌入层:获取词嵌入的层
#嵌入层 其⾏数为词典⼤小（input_dim），列数为每个词向量的维度（output_dim）。设词典⼤小为 20，词向量的维度为 4
embed = nn.Embedding(input_dim=20, output_dim=4)
embed.initialize()
print(embed.weight)
#嵌⼊层  输⼊⼀个词的索引 i，嵌⼊层返回权重矩阵的第 i ⾏作为它的词向量。形状为（2，3）的索引输⼊进嵌⼊层，由于词向量的维度为 4，我们得到形状为（2，3，4）的词向量。
x = nd.array([[1, 2, 3], [4, 5, 6]])
print(embed(x))

#小批量乘法 给定两个形状分别为（n，a，b）和（n，b，c）的 NDArray，小批量乘法输出的形状为（n，a，c）。
X = nd.ones((2, 1, 4))
Y = nd.ones((2, 4, 6))
print(nd.batch_dot(X, Y))

#跳字模型前向计算  跳字模型的输⼊包含中⼼词索引 center 以及连结的背景词与噪⾳词索引 contexts_and_negatives。 这两个变量先通过词嵌⼊层分
# 别由词索引变换为词向量，再通过小批量乘法得到形状为（批量⼤小，1，max_len）的输出  输出中的每个元素是中⼼词向量与背景词向量或噪⾳词向量的内积。
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = nd.batch_dot(v, u.swapaxes(1, 2))
    return pred

#训练模型
#(1)定义损失函数：⼆元交叉熵损失函数
loss = gloss.SigmoidBinaryCrossEntropyLoss()
#自己实现的二元交叉熵损失函数
def sigmd(x):
    return -math.log(1 / (1 + math.exp(-x)))

#(2)初始化模型参数
embed_size = 100  #将超参数词向量维度 embed_size 设置成 100
net = nn.Sequential()
net.add(nn.Embedding(input_dim=len(idx_to_token), output_dim=embed_size), nn.Embedding(input_dim=len(idx_to_token), output_dim=embed_size))

#(3) 训练
def train(net, lr, num_epochs):
    ctx = d2l.try_gpu()
    net.initialize(ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
    for epoch in range(num_epochs):
        start, l_sum, n = time.time(), 0.0, 0
        for batch in data_iter:
            center, context_negative, mask, label = [data.as_in_context(ctx) for data in batch]
            with autograd.record():
                pred = skip_gram(center, context_negative, net[0], net[1])
                # 使⽤掩码变量 mask 来避免填充项对损失函数计算的影响。
                l = (loss(pred.reshape(label.shape), label, mask) * mask.shape[1] / mask.sum(axis=1))
            l.backward()
            trainer.step(batch_size)
            l_sum += l.sum().asscalar()
            n += l.size
        print('epoch %d, loss %.2f, time %.2fs' % (epoch + 1, l_sum / n, time.time() - start))

train(net, 0.005, 5)


#应用词嵌入模型
def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data()
    x = W[token_to_idx[query_token]]
    # 添加的 1e-9 是为了数值稳定性。
    cos = nd.dot(W, x) / (nd.sum(W * W, axis=1) * nd.sum(x * x) + 1e-9).sqrt()
    topk = nd.topk(cos, k=k+1, ret_typ='indices').asnumpy().astype('int32')
    for i in topk[1:]: # 除去输⼊词。
        print('cosine sim=%.3f: %s' % (cos[i].asscalar(), (idx_to_token[i])))

get_similar_tokens('student', 3, net)