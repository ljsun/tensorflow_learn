# encoding = utf-8
import numpy as np
import pickle
from functools import reduce
import os

base_dir = os.path.dirname(__file__)


def build_vocab(data):
    special_words = ['<PAD>', '<GO>', '<EOS>']
    set_words = list(set([word for line in data for word in line]))
    idx_to_vocab = {idx: word for idx, word in enumerate(special_words + set_words)}
    vocab_to_idx = {word: idx for idx, word in idx_to_vocab.items()}

    return idx_to_vocab, vocab_to_idx, len(set_words)


def load_data(source_file, target_file):
    """
    load data and word_id
    :param source_file: input file
    :param target_file: output file
    :return: input and output array with word_id
    """

    source_data = list(open(source_file, mode='r').readlines())
    source_data = [s.strip() for s in source_data]
    target_data = list(open(target_file, mode='r').readlines())
    target_data = [s.strip() for s in target_data]

    # build vocabulary
    idx_to_vocab, vocab_to_idx, vocab_size = build_vocab(source_data + target_data)

    source_data_idx = [[vocab_to_idx[word] for word in line] for line in source_data]
    target_data_idx = [[vocab_to_idx[word] for word in line] + [vocab_to_idx['<EOS>']] for line in target_data]

    # pickle
    pickle.dump(vocab_to_idx, open(os.path.join(base_dir, 'data', 'vocab_to_idx.txt'), mode='wb'))
    pickle.dump(idx_to_vocab, open(os.path.join(base_dir, 'data', 'idx_to_vocab.txt'), mode='wb'))

    return source_data_idx, target_data_idx, vocab_to_idx, vocab_size


def pad_sequence_batch(sequence_batch, pad_idx):
    """
    padding the sequence batch
    :param sequence_batch:
    :param pad_idx:
    :return:
    """
    max_sequence = max(len(sequence) for sequence in sequence_batch)
    return [sequence + (max_sequence - len(sequence)) * [pad_idx] for sequence in sequence_batch], max_sequence


def batch_iter(data, batch_size, num_epochs, pad_idx, shuffle=True):
    """
     generate a batch iterator for data
    :param data: [([x1],[y1]), ([x2],[y2])...]
    :param batch_size:
    :param num_epochs:
    :param pad_idx:
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
            # padding
            data_x_y = data[batch_size * batch_num: batch_size * (batch_num + 1)]
            data_x, data_y = zip(*data_x_y)
            data_x_paded, max_sequence_x = pad_sequence_batch(data_x, pad_idx)
            data_y_paded, max_sequence_y = pad_sequence_batch(data_y, pad_idx)
            data_x_length = [len(x) for x in data_x]
            data_y_length = [len(y) for y in data_y]
            mask_y = [[1]*y_length + [0]*(max_sequence_y - y_length) for y_length in data_y_length]
            mask_y = reduce(lambda a, b: a+b, mask_y)

            yield [data_x_paded, data_y_paded, data_x_length, data_y_length, mask_y]

# if __name__ == '__main__':
#     source_data_idx, target_data_idx, vocab_to_idx, vocab_size = load_data('./data/source.txt', './data/target.txt')
#     train_x, val_x = source_data_idx[: 64], target_data_idx[64:]
#     train_y, val_y = source_data_idx[: 64], target_data_idx[64:]
#     for batch in batch_iter(list(zip(train_x, train_y)), 32, 10, vocab_to_idx['<PAD>']):
#         print(batch[2])