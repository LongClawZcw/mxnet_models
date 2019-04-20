from mxnet import nd
import random

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


# my_seq = list(range(30))
# for X, Y in data_iter_random(my_seq, batch_size=2, num_steps=6):
#     print('X: ', X, '\nY:', Y, '\n')
