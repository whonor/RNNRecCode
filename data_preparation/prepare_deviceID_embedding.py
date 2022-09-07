# -*- coding: utf-8 -*-
# @Time    : 2022/9/5 下午 04:15
# @Author  : HonorWang
# @Email   : honorw@foxmail.com
# @File    : Generate_word_embeddings.py
# @Software: PyCharm
import pickle
import re

import string

import torch
from nltk import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from torchtext.vocab import GloVe
from tqdm import tqdm

import pandas as pd
import numpy as np
import pandas as pd


def word_tokenize(sent: str):
    """ Split sentence into word list using regex.
    Args:
        sent (str): Input sentence
    Return:
        list: word list
    """
    pat = re.compile("[\w]+|[\w]")
    if isinstance(sent, str):
        return pat.findall(sent)
    else:
        return []


# 读取数据
source_train = "Sysetic_data.xlsx"

train_content = pd.read_excel(source_train,
                              usecols=[0, 2, 3, 4, 5],
                              names=['id', 'Seq_device_incident', 'Seq_eventID', 'Seq_incident_description', 'Class'
                                     ])

content = train_content["Seq_device_incident"]

rows = []
for line in content:
    rows.append(word_tokenize(line))
# print(rows)
# 将每个字符串，存入列表中
words = []
for i in rows:
    for j in i:
        words.append(j)
# print(words)

# 将尾部ID单独存入列表
num = []
for w in words:
    num.append(w[-5:])
num = list(set(num))
# 将头部字符单独存入列表
head = []
for w in words:
    head.append(w[:-5])
head = list(set(head))


word_dict_devicehead = {"pad": 0}
word_dict_devicenum = {"pad": 0}
for i, word in enumerate(head):
    word_dict_devicehead[word] = i
for i, word in enumerate(num):
    word_dict_devicenum[word] = i

# 使用glve训练字典中的词向量
word_embedding_dim = 300

if word_embedding_dim == 300:
    glove = GloVe(name='840B', dim=300, cache='../glove', max_vectors=10000000000)
else:
    glove = GloVe(name='6B', dim=word_embedding_dim, cache='../glove', max_vectors=10000000000)
glove_stoi = glove.stoi
glove_vectors = glove.vectors
glove_mean_vector = torch.mean(glove_vectors, dim=0, keepdim=False)

# 生成前部分字符向量
devicehead_word_embedding_vectors = torch.zeros([len(word_dict_devicehead), word_embedding_dim])
for word in tqdm(word_dict_devicehead, desc="embedding...", total=len(word_dict_devicehead)):
    index = word_dict_devicehead[word]
    if index != 0:
        if word in glove_stoi:
            devicehead_word_embedding_vectors[index, :] = glove_vectors[glove_stoi[word]]  # DWAEAAGBT6:
            print("Yes")
        else:
            random_vector = torch.zeros(word_embedding_dim)
            random_vector.normal_(mean=0, std=0.1)
            devicehead_word_embedding_vectors[index, :] = random_vector + glove_mean_vector
            print("No")
print(devicehead_word_embedding_vectors)

print(word_dict_devicehead.keys())

with open("devicehead_word_embedding_vectors.npy", 'wb') as f:
    np.save(f, devicehead_word_embedding_vectors)
with open("deviceEng_dic.pkl", 'wb') as f1:
    pickle.dump(word_dict_devicehead, f1)

# 生成ID数字向量
devicenum_word_embedding_vectors = torch.zeros([len(word_dict_devicenum), word_embedding_dim])
for word in tqdm(word_dict_devicenum, desc="embedding...", total=len(word_dict_devicenum)):
    index = word_dict_devicenum[word]
    if index != 0:
        if word in glove_stoi:
            devicenum_word_embedding_vectors[index, :] = glove_vectors[glove_stoi[word]]  # DWAEAAGBT6:
            print("Yes")
        else:
            random_vector = torch.zeros(word_embedding_dim)
            random_vector.normal_(mean=0, std=0.1)
            devicenum_word_embedding_vectors[index, :] = random_vector + glove_mean_vector
            print("No")

print(devicehead_word_embedding_vectors)

print(word_dict_devicehead.keys())

with open("devicenum_word_embedding_vectors.npy", 'wb') as f2:
    np.save(f2, devicehead_word_embedding_vectors)
with open("deviceNum_dic.pkl", 'wb') as f3:
    pickle.dump(word_dict_devicenum, f3)