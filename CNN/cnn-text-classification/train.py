# encoding = utf-8
import sys
import os

base_dir = os.path.dirname(__file__)
sys.path.append(base_dir)
import tensorflow as tf
import numpy as np

from data_helper import load_data_and_labels, batch_iter
from model import Model
from datetime import datetime
import os

flags = tf.app.flags

flags.DEFINE_float("val_percent", 0.1, "val_percent")
flags.DEFINE_integer("num_classes", 2, "num_classes")

flags.DEFINE_integer("batch_size", 32, "The size of train images [32]")
flags.DEFINE_float("learning_rate", 1e-3, "Learning rate for GradientDesent")
flags.DEFINE_integer("num_epochs", 10, "num_epochs")


flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
flags.DEFINE_integer("embedding_size", 128, "embedding_size")
flags.DEFINE_string("filter_sizes", "3,4,5", "filter_sizes")
flags.DEFINE_integer("num_filters", 128, "num_filters")
flags.DEFINE_float("l2_reg_lambda", 0.3, "l2_reg_lambda")

flags.DEFINE_integer("num_checkpoints", 5, "num_checkpoints")
flags.DEFINE_integer("evaluate_every", 100, "evaluate_every")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")

FLAGS = flags.FLAGS


def train():
    # prepare data
    positive_file = os.path.join(os.path.dirname(__file__), 'data/rt-polaritydata/rt-polarity.pos')
    negative_file = os.path.join(os.path.dirname(__file__), 'data/rt-polaritydata/rt-polarity.neg')
    data_x, data_y, vocab_size = load_data_and_labels(positive_file, negative_file)
    # generate train_data and validate_data
    validate_index = -1 * int(FLAGS.val_percent * len(data_y))
    x_train, x_val = data_x[: validate_index], data_x[validate_index:]
    y_train, y_val = data_y[: validate_index], data_y[validate_index:]
    with tf.Graph().as_default():
        with tf.Session() as sess:
            model = Model(learning_rate=FLAGS.learning_rate,
                          sequence_length=x_train.shape[1],
                          num_classes=FLAGS.num_classes,
                          vocab_size=vocab_size,
                          embedding_size=FLAGS.embedding_size,
                          filter_sizes=list(map(int, FLAGS.filter_sizes.split(','))),
                          num_filters=FLAGS.num_filters,
                          num_checkpoints=FLAGS.num_checkpoints,
                          l2_reg_lambda=FLAGS.l2_reg_lambda
                          )

            # initialize
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)

            def train_op(x_batch, y_batch):
                loss, accuracy, global_step, summaries, _ = sess.run([model.loss, model.accuracy, model.global_step,
                                                                      model.train_summary, model.train_op],
                                                                     feed_dict={
                                                                          model.input_x: x_batch,
                                                                          model.output_y: y_batch,
                                                                          model.dropout: FLAGS.dropout_keep_prob
                                                                        })

                print("step: {:d}, loss {:g}, acc {:g}".format(global_step, loss, accuracy))
                # model.train_summary_writer.add_summary(summaries, global_step)
                return global_step

            def val_op(val_x, val_y):
                loss, accuracy, summaries = sess.run([model.loss, model.accuracy, model.val_summary],
                                                     feed_dict={
                                                         model.input_x: val_x,
                                                         model.output_y: val_y,
                                                         model.dropout: 1.0
                                                      })
                print("loss {:g}, acc {:g}".format(loss, accuracy))
                # model.train_summary_writer.add_summary(val_summary, step)

            # train and validate
            # generate batches
            batches = batch_iter(list(zip(x_train, y_train)), batch_size=FLAGS.batch_size,
                                 num_epochs=FLAGS.num_epochs)

            for batch in batches:
                x_batch, y_batch = zip(*batch)
                x_batch = np.array(x_batch, dtype=np.int32)
                y_batch = np.array(y_batch, dtype=np.int32)
                current_step = train_op(x_batch, y_batch)
                if current_step % FLAGS.evaluate_every == 0:
                    print('Evaluate\n')
                    val_op(val_x=x_val, val_y=y_val)
                # if current_step % FLAGS.checkpoint_every == 0:
                #     path = model.saver.save(sess, model.checkpoint_prefix, global_step=current_step)
                #     print("Saved model checkpoint to {}\n".format(path))


def main(_):
    train()

if __name__ == '__main__':
    tf.app.run()
