from mxnet import nd
import random
import zipfile

def load_data_jay_lyrics():
    with zipfile.ZipFile('../jaychou_lyrics.txt.zip') as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')
    # print(corpus_chars[:40])

    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[0:10000]


    #建立字符索引
    #将数据集⾥⾯所有不同的字符取出来，然后将其逐⼀映射到索引来构造词典。
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    #print(vocab_size)

    #将训练数据集中每个字符转化为索引，并打印前 20 个字符及其对应的索引。
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    # sample = corpus_indices[:20]
    # print('chars:', ''.join([idx_to_char[idx] for idx in sample]))
    # print('indices:', sample)

    # print(corpus_indices,char_to_idx,idx_to_char,vocab_size)
    return corpus_indices,char_to_idx,idx_to_char,vocab_size

# print(load_data_jay_lyrics())
def to_onehot(X,size):
    return [nd.one_hot(x, size) for x in X.T]