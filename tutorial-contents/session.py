"""
Know more, visit my Python tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
tensorflow: 1.1.0
"""
import tensorflow as tf

# construct the calculate graph
m1 = tf.constant([[2, 2]])
m2 = tf.constant([[3], [3]])

dot_operation = tf.matmul(m1, m2)

# exeuute the calculate graph
# method1 use session
session = tf.Session()
result = session.run(dot_operation)
print(result)
session.close()

# method2 use session
with tf.Session() as session:
    result_ = session.run(dot_operation)
    print(result_)