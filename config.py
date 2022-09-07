import argparse
parser = argparse.ArgumentParser()
#长度设置
parser.add_argument('-maxLength', default=10)

#
parser.add_argument('-batch_size', default=64)
parser.add_argument('-epoch', default=5)
parser.add_argument('-lr', default=0.0001)

#模型参数
parser.add_argument('-hidden_size', default=300)
parser.add_argument('-rnn_layers', default=1)
parser.add_argument('-device_hidden', default=150)
parser.add_argument('-event_hidden', default=300)
parser.add_argument('-desc_hidden_size', default=6)


args = parser.parse_args()

