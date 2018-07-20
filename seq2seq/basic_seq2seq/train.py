# encoding = utf-8
import os
import sys

base_dir = os.path.dirname(__file__)
sys.path.append(base_dir)

from functools import reduce
import tensorflow as tf
from data_helper import load_data, batch_iter, pad_sequence_batch
from basic_seq2seq import BasicSeq2Seq

flags = tf.app.flags

flags.DEFINE_float("val_batch_num", 1, "val_percent")

flags.DEFINE_integer("batch_size", 32, "The size of train images [32]")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate for GradientDesent")
flags.DEFINE_integer("num_epochs", 1000, "num_epochs")

flags.DEFINE_integer("embedding_size", 128, "embedding_size")
flags.DEFINE_integer("rnn_size", 128, "rnn_size")
flags.DEFINE_integer("layer_size", 2, "layer_size")

flags.DEFINE_string("checkpoint_path", './run/trained_model.cpkt', "checkpoint_path")

FLAGS = flags.FLAGS


def train():
    # prepare data
    source_file = os.path.join(os.path.dirname(__file__), 'data/source.txt')
    target_file = os.path.join(os.path.dirname(__file__), 'data/target.txt')
    source_data_idx, target_data_idx, vocab_to_idx, vocab_size = load_data(source_file, target_file)

    # generate train_data and validate_data
    val_x, train_x = source_data_idx[: FLAGS.val_batch_num*FLAGS.batch_size], target_data_idx[FLAGS.val_batch_num*FLAGS.batch_size:]
    val_y, train_y = source_data_idx[: FLAGS.val_batch_num*FLAGS.batch_size], target_data_idx[FLAGS.val_batch_num*FLAGS.batch_size:]

    pad_idx = vocab_to_idx['<PAD>']
    go_idx = vocab_to_idx['<GO>']

    # generate training batches
    batches = batch_iter(list(zip(train_x, train_y)),
                         batch_size=FLAGS.batch_size,
                         num_epochs=FLAGS.num_epochs,
                         pad_idx=pad_idx)

    # generate validation data
    val_x_paded, max_sequence_val_x = pad_sequence_batch(val_x, pad_idx)
    val_y_paded, max_sequence_val_y = pad_sequence_batch(val_y, pad_idx)
    val_x_length = [len(x) for x in val_x]
    val_y_length = [len(y) for y in val_y]
    val_mask_y = [[1] * y_length + [0] * (max_sequence_val_y - y_length) for y_length in val_y_length]
    val_mask_y = reduce(lambda a, b: a + b, val_mask_y)

    # build the graph
    g = tf.Graph()
    with g.as_default():
        model = BasicSeq2Seq(rnn_size=FLAGS.rnn_size,
                             layer_size=FLAGS.layer_size,
                             vocab_size=vocab_size,
                             embedding_size=FLAGS.embedding_size,
                             batch_size=FLAGS.batch_size,
                             learning_rate=FLAGS.learning_rate,
                             go_idx=go_idx)

    with tf.Session(graph=g) as sess:
        # initialize
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        # prepare validate data
        val_feed_dict = {model.input_x: val_x_paded,
                         model.target_id: val_y_paded,
                         model.encoder_sequence_length: val_x_length,
                         model.decoder_sequence_length: val_y_length,
                         model.decoder_mask: val_mask_y}

        # beginning training
        step = 1
        for batch in batches:
            data_x_paded, data_y_paded, data_x_length, data_y_length, mask_y = batch
            feed_dict = {model.input_x: data_x_paded,
                         model.target_id: data_y_paded,
                         model.encoder_sequence_length: data_x_length,
                         model.decoder_sequence_length: data_y_length,
                         model.decoder_mask: mask_y}

            if step % 50 == 0:
                cost, _ = sess.run(fetches=[model.cost, model.train_op], feed_dict=feed_dict)
                print('training step: {:d} | training cost: {:.4f}'.format(step, cost))
            if step % 500 == 0:
                print('********************************')
                cost = sess.run(model.cost, feed_dict=val_feed_dict)
                print('validation step: {:d} | validation cost: {:.4f}'.format(step, cost))
                print('********************************')
            step += 1

        # save the model
        saver = tf.train.Saver()
        saver.save(sess, FLAGS.checkpoint_path)
        print("Model Trained and Saved")


if __name__ == '__main__':
    train()
