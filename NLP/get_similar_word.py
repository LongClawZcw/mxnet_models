from mxnet import nd
from mxnet.contrib import text

#contrib.text 包提供的预训练词嵌⼊的名称
print(text.embedding.get_pretrained_file_names().keys())

#给定词嵌⼊名称，我们可以查看该词嵌⼊提供了哪些预训练的模型。
#预训练的 GloVe 模型的命名规范⼤致是“模型.（数据集.）数据集词数. 词向量维度.txt”
#值得⼀提的是，fastText 有预训练的中⽂词向量（pretrained_file_name=‘wiki.zh.vec’）。
print(text.embedding.get_pretrained_file_names('glove'))  #可以使用fasttext代替glove

glove_6b50d = text.embedding.create('glove', pretrained_file_name='glove.6B.50d.txt')
print(len(glove_6b50d))

#通过词来获取它在词典中的索引，也可以通过索引获取词。
print(glove_6b50d.token_to_idx['beautiful'], glove_6b50d.idx_to_token[3367])

#应用预训练词向量   使⽤余弦相似度来搜索近义词
def knn(W, x, k):
    cos = nd.dot(W, x.reshape((-1,))) / ((nd.sum(W * W, axis=1) + 1e-9).sqrt() * nd.sum(x * x).sqrt())
    topk = nd.topk(cos, k=k, ret_typ='indices').asnumpy().astype('int32')
    return topk, [cos[i].asscalar() for i in topk]

def get_similar_tokens(query_token, k, embed):
    topk, cos = knn(embed.idx_to_vec, embed.get_vecs_by_tokens([query_token]), k+1)
    for i, c in zip(topk[1:], cos[1:]): # 除去输⼊词。
        print('cosine sim=%.3f: %s' % (c, (embed.idx_to_token[i])))

get_similar_tokens('chip', 3, glove_6b50d)
get_similar_tokens('baby', 3, glove_6b50d)
get_similar_tokens('beautiful', 3, glove_6b50d)

#求类比词  对于类⽐关系中的四个词 a : b :: c : d，给定前三个词 a、b 和 c，求 d
def get_analogy(token_a, token_b, token_c, embed):
    vecs = embed.get_vecs_by_tokens([token_a, token_b, token_c])
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = knn(embed.idx_to_vec, x, 1)
    return embed.idx_to_token[topk[0]]

print(get_analogy('man', 'woman', 'son', glove_6b50d))
print(get_analogy('beijing', 'china', 'tokyo', glove_6b50d))
print(get_analogy('do', 'did', 'go', glove_6b50d))