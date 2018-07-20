"""
Know more, visit my Python tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
tensorflow: 1.1.0
numpy
"""
import tensorflow as tf
import numpy as np

tf.set_random_seed(1)
np.random.seed(1)

x = np.linspace(-1, 1, 100)[:, np.newaxis]
noise = np.random.normal(0, 0.1, size=x.shape)
y = np.power(x, 2) + noise

with tf.variable_scope('Inputs'):
    tf_x = tf.placeholder(tf.float32, x.shape, name='x')
    tf_y = tf.placeholder(tf.float32, y.shape, name='y')

with tf.variable_scope('Net'):
    l1 = tf.layers.dense(tf_x, 10, tf.nn.relu, name='hidden_layer')
    output = tf.layers.dense(l1, 1, name='output_layer')

    # add to summary
    tf.summary.histogram('h_out', l1)
    tf.summary.histogram('pred', output)

loss = tf.losses.mean_squared_error(tf_y, output, scope='loss')
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)
tf.summary.scalar('loss', loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter('./log', sess.graph)
    # operation to merge all summary
    merge_op = tf.summary.merge_all()

    for step in range(100):
        # train and net output
        _, result = sess.run([train_op, merge_op], {tf_x: x, tf_y: y})
        writer.add_summary(summary=result, global_step=step)
