"""
Know more, visit my Python tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
tensorflow: 1.1.0
matplotlib
numpy
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)

mnist = input_data.read_data_sets('./mnist', one_hot=True)  # they has been normalized to range (0,1)
test_x = mnist.test.images[:2000]
test_y = mnist.test.labels[:2000]


class CNNModel(object):
    def __init__(self, batch_size, learning_rate):
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def build(self):
        with tf.Graph().as_default():
            self.tf_x = tf.placeholder(dtype=tf.float32, shape=[None, 28*28])
            self.tf_y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

            self.reshape_tf_x = tf.reshape(self.tf_x, shape=[-1, 28, 28, 1])
            # conv1
            conv1 = tf.layers.conv2d(inputs=self.reshape_tf_x,
                             filters=16,
                             kernel_size=(5, 5),
                             strides=(1, 1),
                             padding='SAME',
                             activation=tf.nn.relu)
            # pool1
            pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                            pool_size=(2, 2),
                                            strides=(2, 2),
                                            )
            # conv2
            conv2 = tf.layers.conv2d(inputs=pool1,
                                     filters=32,
                                     kernel_size=(5, 5),
                                     strides=(1, 1),
                                     padding='SAME',
                                     activation=tf.nn.relu)
            # pool2
            pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                            pool_size=(2, 2),
                                            strides=(2, 2))
            # flat
            flat = tf.reshape(tensor=pool2, shape=[-1, 7*7*32])
            output = tf.layers.dense(flat, 10)

            self.loss = tf.losses.softmax_cross_entropy(self.tf_y, logits=output)
            self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

            self.accuracy = tf.metrics.accuracy(labels=tf.argmax(self.tf_y, axis=1), predictions=tf.argmax(output, axis=1),)[1]

    def run(self):
        with tf.Session() as sess:
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)

            for step in range(1000):
                b_x, b_y = mnist.train.next_batch(self.batch_size)

                _, train_loss, train_accuracy = sess.run(fetches=[self.train_op, self.loss, self.accuracy],
                                         feed_dict={
                                             self.tf_x: b_x,
                                             self.tf_y: b_y
                                         })
                if step % 50 == 0:
                    test_loss, test_accuracy = sess.run([self.loss, self.accuracy]
                                                        , feed_dict={self.tf_x: test_x, self.tf_y: test_y})
                    print("Step:", step, '|test loss %.4f' % test_loss, '|test accuracy %.4f' % test_accuracy)


flags = tf.app.flags
flags.DEFINE_integer("batch_size", 32, "The size of train images [32]")
flags.DEFINE_float("learning_rate", 0.05, "Learning rate for GradientDesent")
FLAGS = flags.FLAGS


def main(_):
    cnnModel = CNNModel(batch_size=FLAGS.batch_size,
                        learning_rate=FLAGS.learning_rate)
    cnnModel.build()
    cnnModel.run()

if __name__ == '__main__':
    tf.app.run()
