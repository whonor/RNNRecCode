# -*- coding: utf-8 -*-

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np


if __name__ == '__main__':
    source_train = "Sysetic_data.xlsx"
    train_news = pd.read_excel(source_train,
                               usecols=[0, 2, 3, 4, 5],
                               names=['id', 'Seq_device_incident', 'Seq_eventID', 'Seq_incident_description', 'Class'
                               ])
    df = train_news['Seq_incident_description']

    tfidf_model = TfidfVectorizer().fit(df)
    sparse_result = tfidf_model.transform(df)  # 得到tf-idf矩阵，稀疏矩阵表示法

    Seq_incident_description_embeddings = sparse_result.todense()
    # [50, 6]
    print(Seq_incident_description_embeddings)

    # tf_vectors = np.array(df['content'])
    with open('tfidf_descEmbedding.npy', 'wb') as f:
        np.save(f, Seq_incident_description_embeddings)

