"""
？perplexity计算
？关于多层RNN的理解。。。记录在云笔记上
？关lstm如何记录loss次数
"""
# encoding = utf-8
import numpy as np
import tensorflow as tf
from ..models.tutorials.rnn.ptb import reader

"""
会用到的数据集--ptb.test.txt，ptb.train.txt，ptb.valid.txt
参照https://blog.csdn.net/gentelyang/article/details/77451290理解PTB数据集
关于Perplexity理解，参照https://blog.csdn.net/luo123n/article/details/48902815
"""

"""
RNN 简要介绍
循环神经网络中的状态是通过一个向量来表示的，这个向量的维度也称为循环神经网络隐藏层的大小，就是课件中的h
"""

DATA_PATH = '/home/sunlianjie/PycharmProjects/tensorPrac/lstm/data/simple-examples/data'
# 隐藏层规模
HIDDEN_SIZE = 200
NUM_LAYERS = 2
VOCAB_SIZE = 10000

LEARNING_RATE = 1.0
TRAIN_BATCH_SIZE = 20
TRAIN_NUM_STEP = 35

EVAL_BATCH_SIZE = 1
EVAL_NUM_STEP = 1
NUM_EPOCH = 2
KEEP_PROB = 0.5
# 用于控制梯度爆炸的参数
MAX_GRAD_NORM = 5


class PTBInput(object):

    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = reader.ptb_producer(data, batch_size, num_steps, name=name)


# 定义语言模型的class
class PTBModel(object):

    def __init__(self, is_training, batch_size, num_steps):
        # 记录使用的batch大小和截断长度
        self.batch_size = batch_size
        self.num_steps = num_steps

        # 定义输入层
        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])

        # 定义预期输出
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        # 定义使用LSTM结构为循环体结构且使用dropout的深层循环神经网络
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
        if is_training:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_cell, output_keep_prob=KEEP_PROB
            )

        # 使用两层LSTM网络，前一层的LSTM的输出作为后一层的输入
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * NUM_LAYERS)

        # 初始化最初的状态，其实就是初始化lstm中的c和t
        self.initial_state = cell.zero_state(batch_size, tf.float32)
        # 将单词ID转换成单词向量。因为总共有VOCAB_SIZE个单词，每个单词向量的维度为
        # HIDDEN_SIZE， 所以embedding参数的维度为VOCAB x HIDDEN_SIZE
        embedding = tf.get_variable("embedding", [VOCAB_SIZE, HIDDEN_SIZE])

        # 将原本batch_size x num_steps个单词ID转化为单词向量，转化后的输入层维度
        # 为batch_size x num_steps x HIDDEN_SIZE
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        # 只在训练时使用dropout
        if is_training:
            inputs = tf.nn.dropout(inputs, KEEP_PROB)

        # 定义输出列表。先将不同时刻LSTM结构的输出收集起来，再通过全连接层得到最终的输出
        outputs = []
        # state 存储不同batch中LSTM的状态，使其初始化为0。
        state = self.initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                # cell_out: [batch, hidden_size]
                cell_output, state = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        # 把输出队列展开成[batch, hidden_size*num_steps]的形状，然后再
        # reshape成[batch*num_steps, hidden_size]的形状
        output = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])
        # 将从LSTM中得到的输出再经过一个全连接层得到最后的预测结果，最终的预测结果在
        # 每一个时刻上都是一个长度为VOCAB_SIZE的数组，经过softmax层之后表示下一个
        # 位置是不同单词的概率
        weight = tf.get_variable("weight", [HIDDEN_SIZE, VOCAB_SIZE], dtype=tf.float32)
        bias = tf.get_variable("bias", [VOCAB_SIZE], dtype=tf.float32)
        # logits shape [batch*num_steps, VOCAB_SIZE]
        logits = tf.matmul(output, weight) + bias

        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            # targets shape [batch, num_steps]
            [tf.reshape(self.targets, [-1])],
            # 损失的权重
            [tf.ones([batch_size * num_steps], dtype=tf.float32)]
        )
        self.cost = tf.reduce_sum(loss) / batch_size
        self.final_state = state

        # 只在训练时定义反向传播
        if not is_training:
            return

        trainable_variables = tf.trainable_variables()
        # 通过clip_by_global_norm控制梯度的大小，避免梯度爆炸
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.cost, trainable_variables), MAX_GRAD_NORM
        )

        # 定义优化方法
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        # 定义训练op
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))


# 使用给定的模型model在数据data上运行train_op并返回全部数据上的perplexity值
def run_epoch(session, model, data, train_op, output_log):

    # 计算perplexity的辅助变量
    total_costs = 0.0
    iters = 0
    state = session.run(model.initial_state)
    # 使用当前数据训练或测试模型
    for step, (x, y) in enumerate(reader.ptb_iterator(data, model.batch_size, model.num_steps)):
        cost, state, _ = session.run(
            [model.cost, model.final_state, train_op],
            {model.input_data: x, model.targets: y,
             model.initial_state: state}
        )

        # 将不同时刻 不同batch的概率加起来就可以得到perplexity公式右边部分
        total_costs += cost
        iters += model.num_steps

        # 只有在训练时输出日志
        if output_log and step % 100 == 0:
            print("After %d steps, perplexity is %3.f" % (step, np.exp(total_costs / iters)))

    # 返回给定模型在给定数据上的perplexity值
    return np.exp(total_costs / iters)


def main(_):
    # 获取原始数据
    train_data, valid_data, test_data, _ = reader.ptb_raw_data(DATA_PATH)

    # 定义初始化函数
    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    # 定义训练时用的循环神经网络模型
    with tf.variable_scope("language_model", reuse=None, initializer=initializer):
        train_model = PTBModel(True, TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)

    # 定义评测用的循环神经网络模型
    with tf.variable_scope("language_model", reuse=True, initializer=initializer):
        eval_model = PTBModel(False, EVAL_BATCH_SIZE, EVAL_NUM_STEP)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    # config.log_device_placement = True

    with tf.Session(config=config) as session:
        tf.initialize_all_variables().run()

        # 使用训练数据训练模型
        for i in range(NUM_EPOCH):
            print("In iteration: %d" % (i+1))
            # 在所有训练数据上训练循环神经模型
            run_epoch(session, train_model, train_data, train_model.train_op, True)

            # 使用验证数据评测模型效果
            valid_perplexity = run_epoch(
                session, eval_model, valid_data, tf.no_op(), False
            )
            print("Epoch: %d Validation Perplexity: %.3f" % (i + 1, valid_perplexity))

        # 最后使用测试数据测试模型效果
        test_perplexity = run_epoch(session, eval_model, test_data, tf.no_op(), False)
        print("Test Perplexity: %3.f" % test_perplexity)


if __name__ == '__main__':
    tf.app.run()
