train.py运行
部分参数如每个embedding的维度，长度等在config设置
构建词表有：desc_dic.pkl, deviceEng_dic.pkl, deviceEng_Num_dic.pkl event_dic.pkl,(这个是对应的命名）
词向量加载我在代码里设置的是None, 即train.py 55-58行，pytorch可以用代码从numpy直接引入对应的词向量，这里调整一下即可，
对于最后一列输入使用的是最大池化，