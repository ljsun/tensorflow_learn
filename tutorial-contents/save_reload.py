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
import matplotlib.pyplot as plt
import numpy as np

tf.set_random_seed(1)
np.random.seed(1)

# fake data
x = np.linspace(-1, 1, 100)[:, np.newaxis]
noise = np.random.normal(0, 0.1, size=x.shape)
y = np.power(x, 2) + noise


def save():
    print('This is save')
    # build neural network
    tf_x = tf.placeholder(dtype=tf.float32, shape=x.shape)
    tf_y = tf.placeholder(dtype=tf.float32, shape=y.shape)
    l = tf.layers.dense(tf_x, 10, tf.nn.relu)
    o = tf.layers.dense(l, 1)
    loss = tf.losses.mean_squared_error(tf_y, o)
    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # define a saver for saving and restoring
        saver = tf.train.Saver()

        for step in range(100):
            sess.run(train_op, {tf_x: x, tf_y: y})

        saver.save(sess, './params', write_meta_graph=False)

        # plotting
        pred, l = sess.run([o, loss], {tf_x: x, tf_y: y})
        plt.figure(1, figsize=(10, 5))
        plt.subplot(121)
        plt.scatter(x, y)
        plt.plot(x, pred, 'r-', lw=5)
        plt.text(-1, 1.2, 'Save Loss=%.4f' % l, fontdict={'size': 15, 'color': 'red'})


def reload():
    print('This is reload')
    # build entire net again and restore
    # 同样先要构建静态图
    tf_x = tf.placeholder(tf.float32, x.shape)
    tf_y = tf.placeholder(tf.float32, y.shape)
    l_ = tf.layers.dense(tf_x, 10, tf.nn.relu)
    o_ = tf.layers.dense(l_, 1)
    loss_ = tf.losses.mean_squared_error(tf_y, o_)

    sess = tf.Session()
    # don't need to initialize variables, just restoring trained variables
    saver = tf.train.Saver()
    saver.restore(sess, './params')

    # plotting
    pred, l = sess.run([o_, loss_], {tf_x: x, tf_y: y})
    plt.subplot(122)
    plt.scatter(x, y)
    plt.plot(x, pred, 'r-', lw=5)
    plt.text(-1, 1.2, 'Reload Loss=%.4f' % l, fontdict={'size': 15, 'color': 'red'})
    plt.show()

save()

# destroy previous net
tf.reset_default_graph()

reload()