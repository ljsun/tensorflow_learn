"""
参考链接：https://www.jianshu.com/p/aab40f439012
"""

# encoding = utf-8
import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.contrib.seq2seq import *


class BasicAttention(object):
    def __init__(self, rnn_size, num_layers, embedding_size, word_to_idx, beam_search, beam_size, learning_rate, max_gradient_norm, forward_only):
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        self.vocab_size = len(word_to_idx)
        self.word_to_idx = word_to_idx
        self.beam_search = beam_search
        self.beam_size = beam_size
        self.learning_rate = learning_rate
        self.max_gradient_norm = max_gradient_norm
        self.forward_only = forward_only
        # 执行模型构建部分的代码
        self.build_model()

    def _create_rnn_cell(self):
        def single_rnn_cell():
            # 创建单个cell
            single_cell = tf.nn.rnn_cell.LSTMCell(self.rnn_size)
            # 添加dropout
            cell = tf.nn.rnn_cell.DropoutWrapper(single_cell, output_keep_prob=self.keep_prob_placeholder)
            return cell
        # 创建multiRnnCell
        cell = tf.nn.rnn_cell.MultiRNNCell([single_rnn_cell() for _ in range(self.num_layers)])
        return cell

    def build_model(self):
        print("building model... ...")
        # *********************
        # placeholder
        self.encoder_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name='encoder_inputs')
        self.encoder_inputs_length = tf.placeholder(dtype=tf.int32, shape=[None], name='encoder_inputs_length')

        # 因为batch_size与keep_prob_placeholder的值在训练与测试时可能会有不同，因此要定义为placeholder
        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        self.keep_prob_placeholder = tf.placeholder(tf.float32, shape=[], name='keep_prob_placeholder')

        self.decoder_targets = tf.placeholder(tf.int32, shape=[None, None], name='decoder_targets')
        self.decoder_targets_length = tf.placeholder(tf.int32, shape=[None], name='decoder_targets_length')

        self.max_target_sequence_length = tf.reduce_max(self.decoder_targets_length, name='max_target_len')
        self.mask = tf.sequence_mask(self.decoder_targets_length, self.max_target_sequence_length,
                                     dtype=tf.float32, name='masks')


        # *********************
        # encoder
        with tf.variable_scope('encoder'):
            # 创建LSTMCell，两层+dropout
            encoder_cell = self._create_rnn_cell()
            # 创建embedding矩阵，encoder和decoder公用该词向量矩阵
            embedding = tf.get_variable('embedding', [self.vocab_size, self.embedding_size])
            # encoder_inputs_embedded shape [batch_size, sequence_length, embedding_size]
            encoder_inputs_embedded = tf.nn.embedding_lookup(embedding, self.encoder_inputs)
            # encoder_outputs shape [batch_size, sequence_length, rnn_size]
            # encoder_state是一个tuple
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell=encoder_cell,
                                                               inputs=encoder_inputs_embedded,
                                                               sequence_length=self.encoder_inputs_length,
                                                               dtype=tf.float32)

        # **********************
        # decoder
        # **********************
        with tf.variable_scope('decoder'):
            # 准备cell，准备数据参数阶段
            if self.forward_only:
                if self.beam_search:
                    # 若使用beam_search，则需要将encoder的输出复制beam_size份
                    print('forward -- use beam_search decoding...')
                    encoder_outputs = tile_batch(encoder_outputs, multiplier=self.beam_size)
                    encoder_state = nest.map_structure(lambda s: tile_batch(s, self.beam_size), encoder_state)
                    encoder_inputs_length = tile_batch(self.encoder_inputs_length, multiplier=self.beam_size)
                    # 如果使用beam_search，则batch_size = self.batch_size * self.beam_size
                    batch_size = self.batch_size * self.beam_size
                else:
                    print('forward -- not use beam_search decoding...')
                    encoder_inputs_length = self.encoder_inputs_length
                    batch_size = self.batch_size
            else:
                print('not forward...')
                encoder_inputs_length = self.encoder_inputs_length
                batch_size = self.batch_size

            attention_mechanism = BahdanauAttention(num_units=self.rnn_size,
                                                    memory=encoder_outputs,
                                                    memory_sequence_length=encoder_inputs_length)
            # 定义decoder阶段要用的LSTMCell，然后为其封装attention wrapper
            decoder_cell = self._create_rnn_cell()
            attentionwrapper = AttentionWrapper(cell=decoder_cell,
                                                attention_mechanism=attention_mechanism,
                                                attention_layer_size=self.rnn_size,
                                                name='Attention_Wrapper')
            decoder_initial_state = attentionwrapper.zero_state(batch_size=batch_size, dtype=tf.float32).clone(cell_state=encoder_state)
            output_layer = tf.layers.Dense(self.vocab_size,
                                           kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                                           activation=tf.nn.relu)

            # begin decode 阶段
            if self.forward_only:
                # start_tokens shape [batch_size]
                start_tokens = tf.ones([self.batch_size], tf.int32) * self.word_to_idx['<go>']
                end_token = self.word_to_idx['<eos>']
                if self.beam_search:
                    # 对于使用beam_search的时候，它里面包含两项（predicted_ids, beam_search_decoder_outputs)
                    # predicted_ids: [batch_size, decoder_targets_length, beam_size]
                    # beam_search_decoder_outputs: BeamSearchDecoderOutput instance namedtuple(scores, predicted_ids, parent_ids)
                    inference_decoder = BeamSearchDecoder(cell=attentionwrapper,
                                                          embedding=embedding,
                                                          start_tokens=start_tokens,
                                                          end_token=end_token,
                                                          initial_state=decoder_initial_state,
                                                          beam_width=self.beam_size,
                                                          output_layer=output_layer)
                else:
                    # 如果不使用则调用GreedyEmbeddingHelper+BasicDecoder的组合进行贪婪式解码
                    # 对于不使用beam_search的时候，它里面包含两项(rnn_outputs, sample_id)
                    # rnn_outputs: [batch_size, decoder_targets_length, vocab_size]
                    # sample_id: [batch_size, decoder_targets_length]
                    predicting_helper = GreedyEmbeddingHelper(embedding=embedding,
                                                              start_tokens=start_tokens,
                                                              end_token=end_token)
                    inference_decoder = BasicDecoder(cell=attentionwrapper,
                                                     helper=predicting_helper,
                                                     initial_state=decoder_initial_state,
                                                     output_layer=output_layer)
                # 调用dynamic_decoder
                predicting_decoder_outputs, _, _ = dynamic_decode(decoder=inference_decoder,
                                                                  maximum_iterations=10)
                if self.beam_search:
                    self.decoder_predict_decode = predicting_decoder_outputs.predicted_ids
                else:
                    self.decoder_predict_decode = tf.expand_dims(predicting_decoder_outputs.sample_id, -1)

                self.decoder_predict_decode = tf.identity(self.decoder_predict_decode, 'decoder_predict_decode')
            else:
                decoder_targets_length = self.decoder_targets_length
                decoder_inputs = tf.concat(
                    [tf.fill([batch_size, 1], self.word_to_idx['<go>']), self.decoder_targets],
                    axis=1)
                decoder_inputs_embedded = tf.nn.embedding_lookup(embedding, decoder_inputs)
                # 训练阶段使用TrainingHelper+BasicDecoder的组合
                training_helper = TrainingHelper(inputs=decoder_inputs_embedded,
                                                 sequence_length=decoder_targets_length,
                                                 time_major=False,
                                                 name='traing_helper')
                training_decoder = BasicDecoder(cell=attentionwrapper,
                                                helper=training_helper,
                                                initial_state=decoder_initial_state,
                                                output_layer=output_layer)
                # 使用dynamic_decoder进行解码，decoder_outputs是一个namedtuple，里面包含两项(rnn_output, sample_id)
                # rnn_output: [batch_size, decoder_targets_length, vocab_size]，保存decode每个时刻每个单词的概率，可以用来计算loss
                # sample_id: [batch_size, decoder_target_length], tf.int32, 保存最终的编码结果。可以表示最后的答案
                training_decoder_outputs, _, _ = dynamic_decode(decoder=training_decoder,
                                                                impute_finished=True,
                                                                maximum_iterations=self.max_target_sequence_length)

        # *************************
        # train_op and loss
        if not self.forward_only:
            with tf.variable_scope('train_and_loss'):
                # 根据输出计算loss和梯度，并定义进行更新的AdamOptimizer和train_op
                self.decoder_logits_train = tf.identity(training_decoder_outputs.rnn_output)
                # self.decoder_predictidx_train = tf.argmax(self.decoder_logits_train, axis=-1)
                # 使用sequence_loss 计算loss
                self.loss = sequence_loss(self.decoder_logits_train, self.decoder_targets, self.mask)

                # train_op
                """
                clip_by_global_norm到底是干什么的？
                Gradient Clipping的引入是为了处理gradient explosion或者gradients vanishing的问题
                tf.clip_by_global_norm(t_list, clip_norm, use_norm=None, name=None)
                t_list[i]的更新公式如下所示
                t_list[i] * clip_norm / max(global_norm, clip_norm)
                其中，global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))
                global_norm是所有梯度的平方和的平方根。
                """
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                trainable_params = tf.trainable_variables()
                gradients = tf.gradients(self.loss, trainable_params)
                clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
                self.train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))

            # summary
            tf.summary.scalar('loss', self.loss)
            self.summary_op = tf.summary.merge_all()

        # *************************
        # 保存模型
        self.saver = tf.train.Saver()

    # def train(self, sess, batch):
    #     feed_dict = {
    #         self.encoder_inputs: batch['encoder_inputs'],
    #         self.encoder_inputs_length: batch['encoder_inputs_length'],
    #         self.decoder_targets: batch['decoder_targets'],
    #         self.decoder_targets_length: batch['decoder_targets_length'],
    #         self.keep_prob_placeholder: 0.5,
    #         self.batch_size: len(batch.encoder_inputs)
    #     }
    #     loss, summary, _, = sess.run([self.loss, self.summary_op, self.train_op], feed_dict=feed_dict)
    #
    #     return loss, summary
    #
    # def eval(self, sess, batch):
    #     feed_dict = {
    #         self.encoder_inputs: batch['encoder_inputs'],
    #         self.encoder_inputs_length: batch['encoder_inputs_length'],
    #         self.decoder_targets: batch['decoder_targets'],
    #         self.decoder_targets_length: batch['decoder_targets_length'],
    #         self.keep_prob_placeholder: 1.0,
    #         self.batch_size: len(batch.encoder_inputs)
    #     }
    #     loss, summary = sess.run([self.loss, self.summary_op], feed_dict=feed_dict)
    #
    #     return loss, summary
    #
    def infer(self, sess, batch):
        feed_dict = {
            self.encoder_inputs: batch['encoder_inputs'],
            self.encoder_inputs_length: batch['encoder_inputs_length'],
            self.keep_prob_placeholder: 1.0,
            self.batch_size: len(batch.encoder_inputs)
        }
        predict = sess.run([self.decoder_predict_decode], feed_dict=feed_dict)
        return predict
