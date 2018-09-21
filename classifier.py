from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC


def KNN(train_feature_sparse_matrix, train_label_list, test_feature_sparse_matrix):
    clf = KNeighborsClassifier()
    clf.fit(train_feature_sparse_matrix, train_label_list)
    predict_test_label_list = clf.predict(test_feature_sparse_matrix)
    return predict_test_label_list


def SVM(train_feature_sparse_matrix, train_label_list, test_feature_sparse_matrix):
    clf = LinearSVC()
    clf.fit(train_feature_sparse_matrix, train_label_list)
    predict_test_label_list = clf.predict(test_feature_sparse_matrix)
    return predict_test_label_list