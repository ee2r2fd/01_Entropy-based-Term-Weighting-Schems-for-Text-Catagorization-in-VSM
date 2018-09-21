from sklearn.metrics import f1_score

import algorithm as algo
import classifier as clf


def get_test_label_list(train_file_route, test_file_route, algorithm, classifier):
    if algorithm == 'tf_idf':
        train_feature_sparse_matrix, train_label_list = algo.tf_idf(train_file_route)
        test_feature_sparse_matrix, test_label_list = algo.tf_idf(test_file_route)
    if algorithm == 'tf_dc':
        train_feature_sparse_matrix, train_label_list = algo.tf_dc(train_file_route)
        test_feature_sparse_matrix, test_label_list = algo.tf_dc(test_file_route)
    else:
        train_feature_sparse_matrix, train_label_list = algo.tf_bdc(train_file_route)
        test_feature_sparse_matrix, test_label_list = algo.tf_bdc(test_file_route)

    if classifier == 'KNN':
        predict_test_label_list = clf.KNN(train_feature_sparse_matrix, train_label_list, test_feature_sparse_matrix)
    else:
        predict_test_label_list = clf.SVM(train_feature_sparse_matrix, train_label_list, test_feature_sparse_matrix)
    return test_label_list, predict_test_label_list


def evaluate(test_label_list, predict_test_label_list):
    '''
    用于评估分类效果（决定分类效果的因素：词的权重算法的选择及分类器的选择和调参）
    :param test_label_list:
    :param predict_test_label_list:
    :return:
    '''
    Macro_F1 = f1_score(test_label_list, predict_test_label_list, average='macro')
    Micro_F1 = f1_score(test_label_list, predict_test_label_list, average='micro')
    print('Macro_F1:', Macro_F1)
    print('Micro_F1:', Micro_F1)


if __name__ == '__main__':
    train_file_route = './data/Reuters_train.txt'
    test_file_route = './data/Reuters_test.txt'
    test_label_list, predict_test_label_list = get_test_label_list(train_file_route, test_file_route, 'tf_idf', 'SVM')
    evaluate(test_label_list, predict_test_label_list)