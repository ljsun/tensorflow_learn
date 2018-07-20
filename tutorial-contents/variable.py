"""
Know more, visit my Python tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
tensorflow: 1.1.0
"""
# encoding = utf-8

import tensorflow as tf

var = tf.Variable(0)

add_operation = tf.add(var, 1)
update_operation = tf.assign(var, add_operation)

with tf.Session() as sess:
    # once define variables, you have to initialize them

    sess.run(tf.global_variables_initializer())
    for _ in range(3):
        sess.run(update_operation)
        print(var)
        print(sess.run(var))