"""
Know more, visit my Python tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

More information about Dataset: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/docs_src/programmers_guide/datasets.md
"""
import tensorflow as tf
import numpy as np

# load your data or create your data in here
npx = np.random.uniform(-1, 1, size=(1000, 1))
npy = np.power(npx, 2) + np.random.normal(0, 0.1, size=npx.shape)
npx_train, npx_test = np.split(npx, [800])
npy_train, npy_test = np.split(npy, [800])

# use placeholder, later you may need different data, pass the different data into placeholder
tf_x = tf.placeholder(npx_train.dtype, shape=npx_train.shape)
tf_y = tf.placeholder(npy_train.dtype, shape=npy_train.shape)

# create a dataloader
dataset = tf.contrib.data.Dataset.from_tensor_slices((tf_x, tf_y))
dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.batch(32)
# repeat for 3 epochs
dataset = dataset.repeat(3)
iterator = dataset.make_initializable_iterator()

# build network
bx, by = iterator.get_next()
l1 = tf.layers.dense(bx, 10, tf.nn.relu)
out = tf.layers.dense(l1, npy.shape[1])
loss = tf.losses.mean_squared_error(by, out)
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

with tf.Session() as sess:
    sess.run([iterator.initializer, tf.global_variables_initializer()], feed_dict={
        tf_x: npx_train,
        tf_y: npy_train
    })

    for step in range(201):
        try:
            _, train_loss = sess.run([train_op, loss])
            if step % 10 == 0:
                test_loss = sess.run(loss, {bx: npx_test, by: npy_test})
                print('step: %i/200' % step, '|train loss:', train_loss, '|test loss:', test_loss)
        except tf.errors.OutOfRangeError:
            print('Finish the last epoch.')
            break
