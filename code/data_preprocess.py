import math
from collections import Counter
import data_util as du
import pandas as pd
from tqdm import tqdm


def create_word_lib_df(train_file_route):
    '''
    根据训练集建立词库
    :param file_route:
    :return:
    '''
    train_data = du.get_data(train_file_route)
    doc_list = train_data.split('\n')

    doc_num = len(doc_list)  # 文档数
    word_list = []

    for doc in tqdm(doc_list):
        doc_word_list = doc.split('\t')[1].strip().split(' ')
        doc_word_only_list = list(set(doc_word_list))
        word_list.extend(doc_word_only_list)

    word_counter = Counter(word_list)
    word_list = list(word_counter.keys())
    word_count_list = list(word_counter.values())
    idf_list = [math.log(doc_num/x, 2) for x in word_count_list]

    word_lib_df = pd.DataFrame(data={'word': word_list, 'idf': idf_list})
    word_lib_df.set_index('word', inplace=True)
    word_lib_df['index'] = list(range(word_lib_df.shape[0]))
    word_lib_route = './data/word_lib.csv'
    word_lib_df.to_csv(word_lib_route)


if __name__ == '__main__':
    train_file_route = './data/Reuters_train.txt'
    create_word_lib_df(train_file_route)