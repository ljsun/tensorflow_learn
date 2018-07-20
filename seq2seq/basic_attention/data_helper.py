# encoding = utf-8
import os
import pickle
import numpy as np
import nltk

base_dir = os.path.dirname(__file__)
filename = os.path.join(base_dir, 'data', 'dataset-cornell-length10-filter1-vocabSize40000.pkl')
# with open(filename, 'rb') as file:
#     data = pickle.load(file)
#
# print(len(data['id2word']))
# for key, value in data['id2word'].items():
#     print(key, value)

padToken, goToken, eosToken, unknownToken = 0, 1, 2, 3


def load_data(filename):
    """
    加载数据，数据包括word2id、idCount、id2word、trainingSamples
    :param filepath: 文件路径
    :return: 返回word2id、idCount、id2word、trainingSamples
    """
    filepath = os.path.join(base_dir, 'data', 'dataset-cornell-length10-filter1-vocabSize40000.pkl')
    print("Loading dataset from {}".format(filename))
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
        word2id = data['word2id']
        id2word = data['id2word']
        trainingSamples = data['trainingSamples']

    return word2id, id2word, trainingSamples


def pad_sequence_batch(sequence_batch, pad_idx):
    """
    padding the sequence batch
    :param sequence_batch: [[Q1], [Q2], ...]
    :param pad_idx:
    :return:
    """
    max_sequence = max(len(sequence) for sequence in sequence_batch)
    return [sequence + (max_sequence - len(sequence)) * [pad_idx] for sequence in sequence_batch], max_sequence


def add_eos(sequence_batch, eos_idx):
    """
    for target sequence add eos_idx
    :param sequence_batch: [[Q1], [Q2], ...]
    :param eos_idx:
    :return:
    """
    return [sequence + [eos_idx] for sequence in sequence_batch]


def sentence2encode(sentence, word2id):
    """
    将sentence分词，然后转化为id 【现只支持单句话】
    :param sentence: 原始输入
    :param word2id:  word2id
    :return: 转化后的id
    """
    if sentence == '':
        return None
    # 分词
    tokens = nltk.word_tokenize(sentence)
    if len(tokens) > 20:
        return None
    # 将每个单词转化为id
    word_idx = []
    for token in tokens:
        word_idx.append(word2id.get(token, unknownToken))
    return [word_idx]


def batch_iter(data, batch_size, shuffle=True):
    """
    根据batch_size大小生成batch数据
    :param data: [[[Q1], [A1]], [[Q2], [A2]] ... ]
    :param batch_size:
    :param num_epochs:
    :param shuffle:
    :return:
    """
    """
    Q1: 在源代码中source sequence为什么要进行逆序？
    Q2: 并没有对target sequence 加入<eos>字符？
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = data_size // batch_size
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        data = data[shuffle_indices]
    for batch_num in range(num_batches_per_epoch):
        data_x_y = data[batch_size * batch_num: batch_size * (batch_num + 1)]
        data_x, data_y = zip(*data_x_y)

        # add_eos
        data_y = add_eos(data_y, eosToken)

        # padding
        data_x_paded, max_sequence_x = pad_sequence_batch(data_x, padToken)
        data_y_paded, max_sequence_y = pad_sequence_batch(data_y, padToken)
        data_x_length = [len(x) for x in data_x]
        data_y_length = [len(y) for y in data_y]

        encoder_inputs = data_x_paded
        encoder_inputs_length = data_x_length
        decoder_targets = data_y_paded
        decoder_targets_length = data_y_length
        yield {'encoder_inputs': encoder_inputs,
               'encoder_inputs_length': encoder_inputs_length,
               'decoder_targets': decoder_targets,
               'decoder_targets_length': decoder_targets_length}
#
if __name__ == '__main__':
    filename = 'dataset-cornell-length10-filter1-vocabSize40000.pkl'
    word2id, id2word, trainingSamples = load_data(filename)
    for batch in batch_iter(trainingSamples, 32):
        print(batch['decoder_targets'])