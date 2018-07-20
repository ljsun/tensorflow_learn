# encoding = utf-8

import re
import numpy as np
from tensorflow.contrib import learn
import os

base_dir = os.path.dirname(__file__)


def clear_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_file, negative_file, is_training=True):
    """
    load data, split word, word2id
    :param positive_file:
    :param negative_file:
    :param is_training:
    :return:
    """
    positive_data = list(open(positive_file, mode='r').readlines())
    positive_data = [s.strip() for s in positive_data]
    negative_data = list(open(negative_file, mode='r').readlines())
    negative_data = [s.strip() for s in negative_data]

    # apply clear_str
    data_x = positive_data + negative_data
    # positive_data = list(map(clear_str, positive_data))
    # negative_data = list(map(clear_str, negative_data))
    data_x = [clear_str(sent) for sent in data_x]
    # data_x = positive_data + negative_data
    if is_training:
        # build vocabulary
        max_document_length = max([len(x.split(' ')) for x in data_x])

        # 把每一个document变为sequence of word ids
        # 用0进行填充，太长则会被剪切
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length=max_document_length)

        data_x = np.array(list(vocab_processor.fit_transform(data_x)))

        vocab_processor.save(os.path.join(base_dir, 'run/vocab'))
        # generate labels
        positive_labels = [[0, 1] for _ in positive_data]
        negative_labels = [[1, 0] for _ in negative_data]
        data_y = np.concatenate([positive_labels, negative_labels], axis=0)
        # shuffle
        np.random.seed(1)
        shuffle_indices = np.random.permutation(np.arange(len(data_y)))
        data_x = data_x[shuffle_indices]
        data_y = data_y[shuffle_indices]

        vocab_size = len(vocab_processor.vocabulary_)

        return data_x, data_y, vocab_size
    else:
        # load vocabulary
        vocab_processor = learn.preprocessing.VocabularyProcessor.restore(os.path.join(base_dir, 'run/vocab'))
        data_x = np.array(list(vocab_processor.fit_transform(data_x)))
        # generate labels
        positive_labels = [[0, 1] for _ in positive_data]
        negative_labels = [[1, 0] for _ in negative_data]
        data_y = np.concatenate([positive_labels, negative_labels], axis=0)

        vocab_size = len(vocab_processor.vocabulary_)
        return data_x, data_y, vocab_size


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    generate a batch iterator for data
    :param data:
    :param batch_size:
    :param num_epochs:
    :param shuffle:
    :return:
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = data_size // batch_size
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            yield data[batch_size*batch_num: batch_size*(batch_num+1)]

# if __name__ == '__main__':
#     positive_file = os.path.join(os.path.dirname(__file__), 'data/rt-polaritydata/rt-polarity.pos')
#     negative_file = os.path.join(os.path.dirname(__file__), 'data/rt-polaritydata/rt-polarity.neg')
#     load_data_and_labels(positive_file=positive_file, negative_file=negative_file, is_training=False)
