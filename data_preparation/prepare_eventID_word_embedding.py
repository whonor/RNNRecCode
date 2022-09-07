# -*- coding: utf-8 -*-
# @Time    : 2022/9/4 下午 04:38
# @Author  : HonorWang
# @Email   : honorw@foxmail.com
# @File    : prepare_eventID_word_embedding.py
# @Software: PyCharm
import csv
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

stops = set(stopwords.words("english"))
interpunctuations = str.maketrans('', '', string.punctuation)
table = str.maketrans('', '', string.digits)

ps = PorterStemmer()  # 提取词根
wnl = WordNetLemmatizer()  # 词形还原


def filter_stopwords(line: str):
    # line = line.translate(table)  # 过滤数字
    line = word_tokenize(line.translate(interpunctuations))  # 先去除标点符号，再变小写，然后分词
    line = ''.join(word + '\t' for word in line )  # 去除停用词, 提取词干
    return line


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

df = train_content['Seq_eventID']
df = df.fillna(" ")
# print(df)
rows = []
for r in df:
    rows.append(word_tokenize(r))
# print(row)
word_list = []
for i in rows:
    for j in i:
        word_list.append(j)
# print(word_list)

formlist = list(set(word_list))
# print(formlist)

# glove训练词向量
word_embedding_dim = 300

word_dict = {"pad":0}

for i, word in enumerate(formlist):
    word_dict[word] = i

if word_embedding_dim == 300:
    glove = GloVe(name='840B', dim=300, cache='../glove', max_vectors=10000000000)
else:
    glove = GloVe(name='6B', dim=word_embedding_dim, cache='../glove', max_vectors=10000000000)
glove_stoi = glove.stoi
glove_vectors = glove.vectors
glove_mean_vector = torch.mean(glove_vectors, dim=0, keepdim=False)
word_embedding_vectors = torch.zeros([len(word_dict), word_embedding_dim])
for word in tqdm(word_dict, desc="embedding...", total=len(word_dict)):
    index = word_dict[word]
    if index != 0:
        if word in glove_stoi:
            word_embedding_vectors[index, :] = glove_vectors[glove_stoi[word]]  # DWAEAAGBT6:
            print("Yes")
        else:
            random_vector = torch.zeros(word_embedding_dim)
            random_vector.normal_(mean=0, std=0.1)
            word_embedding_vectors[index, :] = random_vector + glove_mean_vector
            print("No")
#
# # 生成最后的词向量矩阵
# max_length_per_line = 10
# doc_size = 50
#
# Seq_eventID_word_embeddings = torch.zeros([doc_size, max_length_per_line, word_embedding_dim])
#
# word_embedding = torch.zeros([max_length_per_line, word_embedding_dim])
#
#
# for line, l in enumerate(rows):
#     repeated_times = max_length_per_line - len(l)
#     assert repeated_times >= 0
#     l = l + [0] * repeated_times
#     for index, w in enumerate(l):
#         if w != 0:
#             word_embedding[index, :] = word_embedding_vectors[word_dict[w]]
#
#     Seq_eventID_word_embeddings[line, :, :] = word_embedding
#
# # 最后输出 word_embeddings [50, max_length_per_line, 300]
# print(Seq_eventID_word_embeddings)

with open("eventID_word_embedding_vectors.npy", 'wb') as f:
    np.save(f, word_embedding_vectors)
with open("event_dic.pkl", 'wb') as f1:
    pickle.dump(word_dict, f1)



