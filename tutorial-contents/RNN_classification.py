"""
Know more, visit my Python tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
tensorflow: 1.1.0
matplotlib
numpy
"""
# encoding = utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 64, "The size of train images [32]")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate for GradientDesent")
flags.DEFINE_integer("time_step", 28, "rnn time step")
flags.DEFINE_integer("input_size", 28, "rnn input size")
FLAGS = flags.FLAGS

# plot one examples
mnist = input_data.read_data_sets('./MNIST', one_hot=True)
text_x = mnist.test.images[:2000]
text_y = mnist.test.images[:2000]

# plot one examples
# print(mnist.train.images.shape)
# print(mnist.train.labels.shape)
# plt.imshow(mnist.train.images[0].reshape((28, 28)), cmap='gray')
# plt.title('%i' % np.argmax(mnist.train.labels[0]))
# plt.show()


class RNN(object):
    def __init__(self, batch_size, learning_rate, time_step, input_size):

        self.batch_size = batch_size
        self.g = tf.Graph()
        with self.g.as_default():
            self.input_x = tf.placeholder(dtype=tf.float32, shape=[batch_size, 784])
            self.output_y = tf.placeholder(dtype=tf.int32, shape=[batch_size, 10])

            # reshape
            images = tf.reshape(self.input_x, shape=[batch_size, time_step, input_size])

            # num_units is the dimension of state c and state h
            rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=64)

            outputs, (h_c, h_h) = tf.nn.dynamic_rnn(rnn_cell,
                                                    images,
                                                    dtype=tf.float32)
            output = tf.layers.dense(outputs[:, -1, :], units=10)

            losses = tf.losses.softmax_cross_entropy(onehot_labels=self.output_y, logits=output)
            self.loss = tf.reduce_mean(losses)

            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

            self.accuracy = tf.metrics.accuracy(labels=tf.argmax(self.output_y, axis=1),
                                                predictions=tf.argmax(output, axis=1))[1]

    def run(self):
        with tf.Session(graph=self.g) as sess:
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)

            for step in range(1200):
                b_x, b_y = mnist.train.next_batch(batch_size=self.batch_size)
                _, loss_ = sess.run([self.train_op, self.loss], feed_dict={self.input_x: b_x, self.output_y: b_y})
                if step % 100 == 0:
                    accuracy_ = sess.run(self.accuracy, feed_dict={self.input_x: b_x, self.output_y: b_y})
                    print('train_loss: %.4f' % loss_, "| test accuracy: %.2f" % accuracy_)

if __name__ == '__main__':
    RNN(batch_size=FLAGS.batch_size,
        learning_rate=FLAGS.learning_rate,
        time_step=FLAGS.time_step,
        input_size=FLAGS.input_size).run()
