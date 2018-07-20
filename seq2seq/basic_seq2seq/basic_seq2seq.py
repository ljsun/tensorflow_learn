"""
inference: https://blog.csdn.net/thriving_fcl/article/details/74165062
"""
# encoding = utf-8
import tensorflow as tf
from tensorflow.contrib.seq2seq import *
from tensorflow.python.layers.core import Dense


class BasicSeq2Seq(object):
    def __init__(self, rnn_size, layer_size, vocab_size, embedding_size, batch_size, learning_rate, go_idx):
        self.input_x = tf.placeholder(tf.int32, shape=[batch_size, None], name='input_x')
        self.target_id = tf.placeholder(tf.int32, shape=[batch_size, None], name='target_id')
        self.encoder_sequence_length = tf.placeholder(tf.int32, shape=[batch_size], name='encoder_sequence_length')
        self.decoder_sequence_length = tf.placeholder(tf.int32, shape=[batch_size], name='decoder_sequence_length')
        self.decoder_mask = tf.placeholder(tf.float32, shape=[None], name='decoder_mask')

        with tf.variable_scope('embedding'):
            embedding_matrix = tf.Variable(initial_value=tf.truncated_normal([vocab_size, embedding_size]),
                                           dtype=tf.float32, name='embedding')
            input_x_embedded = tf.nn.embedding_lookup(embedding_matrix, self.input_x)

        # encoder
        with tf.variable_scope('encoder'):
            # lstm cell + dynamic rnn
            encoder = self._get_simple_lstm(rnn_size, layer_size)
            """
            dynamic所对应的执行流程，如执行长度等因素
            """
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder, input_x_embedded,
                                                               sequence_length=self.encoder_sequence_length,
                                                               dtype=tf.float32)
        with tf.variable_scope('decoder_part1'):
            # lstm cell
            cell = self._get_simple_lstm(rnn_size, layer_size)
            # output 全连层
            output_fc_layer = Dense(units=vocab_size,
                                    kernel_initializer=tf.truncated_normal_initializer(mean=0.1, stddev=0.1),
                                    activation=tf.nn.relu)
        # training decoder
        with tf.variable_scope('decoder_part2'):
            decoder_input = tf.concat([tf.constant(go_idx, shape=[batch_size, 1]), self.target_id], axis=1)
            target_id_embedded = tf.nn.embedding_lookup(embedding_matrix, decoder_input)
            training_helper = TrainingHelper(target_id_embedded, self.decoder_sequence_length)
            training_decoder = BasicDecoder(cell=cell,
                                            helper=training_helper,
                                            initial_state=encoder_state,
                                            output_layer=output_fc_layer)
            # decoder_outputs shape [batch_size, time_step, vocab]
            # decoder_final_state shape ？
            # 当impute_finished=True or False时所对应的shape
            # decoder_final_sequence_length shape ？
            training_decoder_outputs, training_decoder_final_state, training_decoder_final_sequence_length = \
                dynamic_decode(training_decoder, impute_finished=True)
            training_decoder_outputs = training_decoder_outputs.rnn_output
        # predicting decoder
        with tf.variable_scope('decoder_part2', reuse=True):
            self.start_tokens = tf.placeholder(dtype=tf.int32, shape=[batch_size], name='start_tokens')
            self.end_token = tf.placeholder(dtype=tf.int32, shape=(), name='end_token')
            predicting_helper = GreedyEmbeddingHelper(embedding=embedding_matrix, start_tokens=self.start_tokens,
                                                      end_token=self.end_token)
            predicting_decoder = BasicDecoder(cell=cell,
                                              helper=predicting_helper,
                                              initial_state=encoder_state,
                                              output_layer=output_fc_layer)
            predicting_decoder_outputs, predicting_decoder_final_state, predicting_decoder_final_sequence_length = \
                dynamic_decode(predicting_decoder, impute_finished=True)
            predicting_logits = tf.identity(predicting_decoder_outputs.sample_id, name='predictions')

        # loss
        with tf.name_scope('loss'):
            # one_hot
            target_id_reshaped_one_hot = tf.one_hot(tf.reshape(self.target_id, [-1]), vocab_size)
            training_decoder_outputs = tf.reshape(training_decoder_outputs, [-1, vocab_size])
            # self.cost shape []
            # self.cost = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_id_reshaped,
            #                                                            logits=self.training_decoder_outputs)
            # self.cost = self.cost * self.decoder_mask
            softmax = tf.nn.softmax(training_decoder_outputs, dim=-1)

            self.cost = tf.reduce_sum(-tf.multiply(target_id_reshaped_one_hot, tf.log(softmax)), axis=1)
            self.cost = tf.reduce_sum(self.cost * self.decoder_mask)
            self.cost = self.cost / tf.cast(tf.reduce_sum(self.decoder_sequence_length), dtype=tf.float32)

        with tf.name_scope('train_op'):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = optimizer.minimize(self.cost)

    def _get_simple_lstm(self, rnn_size, layer_size):
        lstm_layers = [tf.nn.rnn_cell.LSTMCell(rnn_size) for _ in range(layer_size)]
        return tf.nn.rnn_cell.MultiRNNCell(lstm_layers)
