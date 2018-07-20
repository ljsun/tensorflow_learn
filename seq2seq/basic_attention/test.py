# encoding = utf-8
import os
import sys

base_dir = os.path.dirname(__file__)
sys.path.append(base_dir)

import tensorflow as tf
from tensorflow.contrib.seq2seq import *
from tensorflow.python import pywrap_tensorflow
import numpy as np
from data_helper import load_data, sentence2encode, batch_iter
from basic_attention import BasicAttention


# tf.app.flags.DEFINE_integer('rnn_size', 1024, 'Number of hidden units in each layer')
# tf.app.flags.DEFINE_integer('num_layers', 2, 'Number of layers in each encoder and decoder')
# tf.app.flags.DEFINE_integer('embedding_size', 1024, 'Embedding dimensions of encoder and decoder inputs')
#
# tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate')
# tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size')
# tf.app.flags.DEFINE_integer('numEpochs', 1, 'Number of training epochs')
# tf.app.flags.DEFINE_integer('steps_per_checkpoint', 100, 'Save model checkpoint every this iteration')
# tf.app.flags.DEFINE_string('model_dir', 'model/', 'Path to save model checkpoints')
# tf.app.flags.DEFINE_string('model_name', 'chatbot.ckpt', 'File name used for model checkpoints')

# ************************************************************************************************
tf.app.flags.DEFINE_integer('rnn_size', 512, 'Number of hidden units in each layer')
tf.app.flags.DEFINE_integer('num_layers', 3, 'Number of layers in each encoder and decoder')
tf.app.flags.DEFINE_integer('embedding_size', 512, 'Embedding dimensions of encoder and decoder inputs')

tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
tf.app.flags.DEFINE_integer('batch_size', 256, 'Batch size')
tf.app.flags.DEFINE_integer('numEpochs', 30, 'Number of training epochs')

tf.app.flags.DEFINE_integer('steps_per_checkpoint', 100, 'Save model checkpoint every this iteration')
tf.app.flags.DEFINE_string('model_dir', 'model_two/', 'Path to save model checkpoints')
tf.app.flags.DEFINE_string('model_name', 'chatbot.ckpt', 'File name used for model checkpoints')
FLAGS = tf.app.flags.FLAGS

filename = 'dataset-cornell-length10-filter1-vocabSize40000.pkl'
word2id, id2word, trainingSamples = load_data(filename=filename)


def predict_idx_to_sen_beam(predict_idx, id2word, beam_size):
    """
    将预测的id转化为word
    :param predict_idx: [batch_size, decoder_length, beam_size] or [batch_size, decoder_length]
    :param id2word: id2word
    :param beam_size: beam_size
    :return:
    """
    for single_predict in predict_idx:
        for i in range(beam_size):
            predict_list = np.ndarray.tolist(single_predict[:, :, i])
            predict_seq = [id2word[idx] for idx in predict_list[0] if idx != 2 and idx != -1]

            print(" ".join(predict_seq))


def predict_idx_to_sen(predict_idx, id2word):
    sentences = []
    for single_predict in predict_idx:
        for idx in single_predict[0]:
            sentences.append(id2word[idx[0]])
    print(' '.join(sentences[:-1]))

g = tf.Graph()
with g.as_default():
    beam_search = True
    model = BasicAttention(FLAGS.rnn_size,
                           FLAGS.num_layers,
                           FLAGS.embedding_size,
                           word_to_idx=word2id,
                           beam_search=beam_search,
                           beam_size=5,
                           max_gradient_norm=5.0,
                           learning_rate=FLAGS.learning_rate,
                           forward_only=True)
with tf.Session(graph=g) as sess:
    # 如果存在已经保存的模型的话，就继续训练，否则，就重新开始
    # 从“检查点”文件目录返回CheckpointState proto
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    # ckpt.model_checkpoint_path，获得最近一次保存的检查点文件的路径
    # 关于CheckpointState proto的理解，看checkpoint文件应该就能相对理解
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Reloading model parameters..')
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        raise ValueError('No such file:[{}]'.format(FLAGS.model_dir))
    if beam_search:
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
            # 目前只支持一次输入一句话
            idx = sentence2encode(sentence, word2id)
            feed_dict = {
                model.encoder_inputs: idx,
                model.encoder_inputs_length: [len(idx[0])],
                model.keep_prob_placeholder: 1.0,
                model.batch_size: len(idx)
            }
            print("Replies -------------------------->")
            predict = sess.run([model.decoder_predict_decode], feed_dict=feed_dict)
            predict_idx_to_sen_beam(predict, id2word, beam_size=5)
            sys.stdout.write("> ")
            sys.stdout.flush()
            sentence = sys.stdin.readline()

    # print("> ", "")
    # sys.stdout.flush()
    # sentence = sys.stdin.readline()
    # if not beam_search:
    #     predict_idx_to_sen(predict, id2word)
    # else:
    #     print('using beam search')
    #     sys.stdout.write("> ")
    #     sys.stdout.flush()
    #
    #     predict_idx_to_sen_beam(predict, id2word, beam_size=5)

