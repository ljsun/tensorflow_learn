# encoding = utf-8
import os
import sys

base_dir = os.path.dirname(__file__)
sys.path.append(base_dir)

import tensorflow as tf
from data_helper import load_data, batch_iter
from basic_attention import BasicAttention
from tqdm import tqdm
import math

# tf.app.flags.DEFINE_integer('rnn_size', 1024, 'Number of hidden units in each layer')
# tf.app.flags.DEFINE_integer('num_layers', 2, 'Number of layers in each encoder and decoder')
# tf.app.flags.DEFINE_integer('embedding_size', 1024, 'Embedding dimensions of encoder and decoder inputs')
#
# tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate')
# tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size')
# tf.app.flags.DEFINE_integer('numEpochs', 30, 'Number of training epochs')
#
# tf.app.flags.DEFINE_integer('steps_per_checkpoint', 100, 'Save model checkpoint every this iteration')
# tf.app.flags.DEFINE_string('model_dir', 'model/', 'Path to save model checkpoints')
# tf.app.flags.DEFINE_string('model_name', 'chatbot.ckpt', 'File name used for model checkpoints')

# ************************************************
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


def train(model, sess, batch):
    feed_dict = {
        model.encoder_inputs: batch['encoder_inputs'],
        model.encoder_inputs_length: batch['encoder_inputs_length'],
        model.decoder_targets: batch['decoder_targets'],
        model.decoder_targets_length: batch['decoder_targets_length'],
        model.keep_prob_placeholder: 0.5,
        model.batch_size: len(batch['encoder_inputs'])
    }
    loss, summary, _, = sess.run([model.loss, model.summary_op, model.train_op], feed_dict=feed_dict)

    return loss, summary


def eval(model, sess, batch):
    feed_dict = {
        model.encoder_inputs: batch['encoder_inputs'],
        model.encoder_inputs_length: batch['encoder_inputs_length'],
        model.decoder_targets: batch['decoder_targets'],
        model.decoder_targets_length: batch['decoder_targets_length'],
        model.keep_prob_placeholder: 1.0,
        model.batch_size: len(batch['encoder_inputs'])
    }
    loss, summary = sess.run([model.loss, model.summary_op], feed_dict=feed_dict)

    return loss, summary

# build the graph
g = tf.Graph()
with g.as_default():
    model = BasicAttention(FLAGS.rnn_size,
                           FLAGS.num_layers,
                           FLAGS.embedding_size,
                           word_to_idx=word2id,
                           beam_search=True,
                           beam_size=5,
                           max_gradient_norm=5.0,
                           learning_rate=FLAGS.learning_rate,
                           forward_only=False)
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
        print('Creating new model parameters..')
        sess.run(tf.global_variables_initializer())

    current_step = 0
    summary_writer = tf.summary.FileWriter(FLAGS.model_dir, graph=sess.graph)

    for e in range(FLAGS.numEpochs):
        print("----- Epoch {}/{} -----".format(e + 1, FLAGS.numEpochs))
        batchs = batch_iter(trainingSamples, FLAGS.batch_size)
        for batch in tqdm(batchs, desc='Training'):
            loss, summary = train(model, sess, batch)
            current_step += 1
            # 每进行 # 步保存一次模型
            if current_step % FLAGS.steps_per_checkpoint == 0:
                # perplexity？
                perplexity = math.exp(float(loss)) if loss < 300 else float('inf')
                tqdm.write("----- Step %d -- Loss %.2f -- Perplexity %.2f" % (current_step, loss, perplexity))
                summary_writer.add_summary(summary, current_step)
                checkpoint_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)
                model.saver.save(sess, checkpoint_path, global_step=current_step)
