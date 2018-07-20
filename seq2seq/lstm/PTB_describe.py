"""
PTB 数据集简单描述
"""
from ..models.tutorials.rnn.ptb import reader

# 存放数据的原始路径
DATA_PATH = "/home/sunlianjie/PycharmProjects/tensorPrac/lstm/data/simple-examples/data"

train_data, valid_data, test_data, _ = reader.ptb_raw_data(DATA_PATH)
# 读取数据原始数据
print(len(train_data))
print(train_data[:100])

# 将训练数据组织成batch大小为4、截断长度为5的数据组
result = reader.ptb_iterator(train_data, 4, 5)
# 读取第一个batch中的数据，其中包括每个时刻的输入和对应的正确输出
x, y = result.next()
print("X:{}".format(x))
print("Y:{}".format(y))
