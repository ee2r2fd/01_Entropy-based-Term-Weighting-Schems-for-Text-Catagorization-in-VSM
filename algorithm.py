from collections import Counter

import math
import numpy as np

import data_util as du

from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


def tf_idf(file_route):
    '''
    KNN:
    正确率：0.70
    Macro_F1: 0.6809592503036148
    Micro_F1: 0.7834627683873915
    SVM:
    Macro_F1: 0.3034147735580599
    Micro_F1: 0.7880310644129741
    :param file_route:
    :return:
    '''
    data = du.get_data(file_route)
    doc_list = data.split('\n')

    doc_num = len(doc_list)
    word_lib_df = du.get_word_lib_df()
    word_lib_num = word_lib_df.shape[0]

    label_list = []
    tf_idf = []
    col = []
    row = []
    doc_no = 0

    for doc in tqdm(doc_list):
        label_list.append(doc.split('\t')[0])
        doc_word_list = doc.split('\t')[1].strip().split(' ')
        doc_word_counter = Counter(doc_word_list)
        doc_lenth = len(doc_word_list)
        for word, count in doc_word_counter.items():
            try:
                one_word_tf = count/doc_lenth
                one_word_idf = word_lib_df.loc[word, 'idf']  # 对于测试集有可能出现异常，即测试集中的某个词在词库中查不到因为词库是训练集构造的
                # fixme 用训练集构造的词库中报错说没有训练集第4543个doc中的nan这个词而实际上词库中有这个词 KeyError: 'the label [nan] is not in the [index]'
                one_word_tf_idf = one_word_tf*one_word_idf
                row.append(doc_no)
                col.append(word_lib_df.loc[word, 'index'])
                tf_idf.append(one_word_tf_idf)
            except:
                continue
        doc_no += 1
    tf_idf_train_feature_sparse_matrix = csr_matrix((tf_idf, (row, col)), shape=(doc_num, word_lib_num))
    # print(len(train_label_list), train_label_list[0])
    le = LabelEncoder()
    le.fit(label_list)
    # print(le.classes_)  # 显示出共八种标签['acq' 'crude' 'earn' 'grain' 'interest' 'money-fx' 'ship' 'trade']
    train_label_list = list(le.transform(label_list))
    # print(train_label_list[0: 5])  # 显示训练集中前五个文档的被编码后的标签[2, 0, 2, 2, 2]
    return tf_idf_train_feature_sparse_matrix, train_label_list


def tf_dc(file_route):
    '''
    KNN:
    正确率：0.787
    Macro_F1: 0.6607617686270945
    Micro_F1: 0.7866605756052993
    :param file_route:
    :return:
    '''
    data = du.get_data(file_route)
    doc_list = data.split('\n')

    word_lib_df = du.get_word_lib_df()
    label_list = ['acq', 'crude', 'earn', 'grain', 'interest', 'money-fx', 'ship', 'trade']
    word_lib_df['doc_num'] = [0]*word_lib_df.shape[0]  # 初始化f(t)
    for label in label_list:
        word_lib_df[label] = [0] * word_lib_df.shape[0]  # 初始化f(t, c)

    for doc in tqdm(doc_list):
        doc_label = doc.split('\t')[0]
        doc_word_list = doc.split('\t')[1].strip().split(' ')
        doc_word_counter = Counter(doc_word_list)
        for word in doc_word_counter.keys():
            try:
                word_lib_df.loc[word, 'doc_num'] += 1  # 给f(t)赋值
                word_lib_df.loc[word, doc_label] += 1  # 给f(t, c)赋值
            except:
                continue

    # print(word_lib_df.head())

    doc_label_list = []
    row = []
    col = []
    value = []
    doc_no = 0
    for doc in tqdm(doc_list):
        doc_label = doc.split('\t')[0]
        doc_label_list.append(doc_label)
        doc_word_list = doc.split('\t')[1].strip().split(' ')
        doc_length = len(doc_word_list)
        doc_word_counter = Counter(doc_word_list)
        for word, count in doc_word_counter.items():
            try:
                one_word_tf = count/doc_length
                fen_zi = 0
                for label in label_list:
                    if word_lib_df.loc[word, label] != 0:
                        temp = word_lib_df.loc[word, label] / word_lib_df.loc[word, 'doc_num']
                        fen_zi += temp*math.log(temp, 2)
                    else:
                        continue
                one_word_dc = 1 + fen_zi / math.log(len(label_list), 2)
                one_word_tf_dc = one_word_tf * one_word_dc
                # print(word+':', one_word_tf_dc)
                row.append(doc_no)
                col.append(word_lib_df.loc[word, 'index'])
                value.append(one_word_tf_dc)
            except:
                continue
        doc_no += 1
    tf_dc_feature_sparse_matrix = csr_matrix((value, (row, col)), shape=(len(doc_label_list), word_lib_df.shape[0]))
    le = LabelEncoder()
    le.fit(doc_label_list)

    return tf_dc_feature_sparse_matrix, le.transform(doc_label_list)


def tf_bdc(file_route):
    '''
    KNN:
    正确率：0.81
    Macro_F1: 0.6809592503036148
    Micro_F1: 0.7834627683873915
    :param file_route:
    :return:
    '''
    data = du.get_data(file_route)
    doc_list = data.split('\n')
    label_list = []

    for doc in tqdm(doc_list):
        doc_label = doc.split('\t')[0]
        label_list.append(doc_label)

    label_list_counter = Counter(label_list)  # 得到f(c)
    one_label_list = label_list_counter.keys()  # ['earn', 'acq', 'trade', 'ship', 'grain', 'crude', 'interest', 'money-fx']

    word_lib_df = du.get_word_lib_df()
    for label in one_label_list:
        word_lib_df[label] = [0]*word_lib_df.shape[0]  # 初始化f(t, c)

    for doc in tqdm(doc_list):
        doc_label = doc.split('\t')[0]
        doc_word_list = doc.split('\t')[1].strip().split(' ')
        doc_word_counter = Counter(doc_word_list)
        for word in doc_word_counter.keys():
            try:
                word_lib_df.loc[word, doc_label] += 1  # 得到f(t, c)
            except:
                continue

    row = []
    col = []
    value = []
    doc_no = 0
    for doc in tqdm(doc_list):
        doc_word_list = doc.split('\t')[1].split(' ')
        doc_word_counter = Counter(doc_word_list)
        for word, word_count in doc_word_counter.items():
            try:
                fenzi = 0
                for label1, label_count1 in label_list_counter.items():
                    if word_lib_df.loc[word, label1] != 0:
                        fen_zi = word_lib_df.loc[word, label1]/label_count1
                        fen_mu = 0
                        for label2, label_count2 in label_list_counter.items():
                            if word_lib_df.loc[word, label2] != 0:
                                fen_mu += word_lib_df.loc[word, label2]/label_count2
                            else:
                                continue
                        temp = fen_zi/fen_mu
                        fenzi += temp*math.log(temp, 2)
                    else:
                        continue
                bdc = 1 + fenzi/math.log(len(label_list_counter), 2)
                tf = word_count/len(doc_word_list)
                tf_bdc = tf*bdc
                # print(word+':', tf_bdc)
                row.append(doc_no)
                col.append(word_lib_df.loc[word, 'index'])
                value.append(tf_bdc)
            except:
                continue
        doc_no += 1
    tf_bd_feature_sparse_matrix = csr_matrix((value, (row, col)), shape=(len(doc_list), word_lib_df.shape[0]))
    le = LabelEncoder()
    le.fit(label_list)

    return tf_bd_feature_sparse_matrix, le.transform(label_list)

















