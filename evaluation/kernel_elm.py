import numpy as np
from sklearn.metrics import pairwise
from collections import Counter


class ELM:
    def __init__(self, gender_vec = None, c=1, weighted=False, kernel='rbf', deg=3,
                 is_classification=False, gender_aware=False):
        super(self.__class__, self).__init__()

        assert kernel in ["rbf", "linear", "poly", "sigmoid"]
        self.x_train = []
        self.C = c
        self.weighted = weighted
        self.gender_aware = gender_aware
        self.beta = []
        self.kernel = kernel
        self.is_classification = is_classification
        self.deg = deg
        self.gender_vec = gender_vec

    def fit(self, x_train, y_train, gender_vec=None):
        """
        Calculate beta using kelm_kernel.
        :param x_train: features of train set
        :param y_train: labels of train set
        :param gender_vec: gender labels
        :return:
        """

        if gender_vec is None:
            gender_vec = []
        self.x_train = x_train
        n = len(x_train)

        if self.is_classification:
            if len(y_train.shape) == 2:
                class_num = 4
                y_one_hot = np.eye(class_num)[y_train]
                y_one_hot = np.transpose(y_one_hot, axes=(2,0,1))

            elif len(np.unique(np.array(y_train))) == 2:
                class_num = 2
                y_one_hot = np.eye(class_num)[y_train]

            else:
                class_num = 25
                y_one_hot = np.eye(class_num)[y_train]

        else:
            y_one_hot = y_train

        if self.kernel == 'rbf':
            kernel_func = pairwise.rbf_kernel(x_train)
        elif self.kernel == 'poly':
            kernel_func = pairwise.polynomial_kernel(x_train, degree=self.deg)
        elif self.kernel == 'sigmoid':
            kernel_func = pairwise.sigmoid_kernel(x_train)
        elif self.kernel == 'linear':
            kernel_func = pairwise.linear_kernel(x_train)

        # Not using weights at the moment
        if self.is_classification and self.weighted and len(y_train.shape) != 2:
            W = np.zeros((n, n))

            hist = np.array([list(y_train).count(i) for i in range(class_num)])
            hist = 1/ hist

            for i in range(len(y_train)):
                W[i, i] = hist[int(y_train.values[i])]

            beta = np.matmul(np.linalg.inv(np.matmul(W, kernel_func) +
                                           np.identity(n) / np.float32(self.C)), np.matmul(W, y_one_hot))

        if self.is_classification and self.gender_aware:
            W_gender = np.zeros((n, n))
            hist_gender = np.array([list(gender_vec).count(i) for i in range(class_num)])
            hist_gender = 1/ hist_gender

            for i in range(len(y_train)):
                W_gender[i, i] = hist_gender[int(gender_vec.values[i])]

            beta = np.matmul(np.linalg.inv(np.matmul(W_gender, kernel_func) +
                                           np.identity(n) / np.float32(self.C)), np.matmul(W_gender, y_one_hot))


        else:
            beta = np.matmul(np.linalg.inv(kernel_func + np.identity(n) / np.float32(self.C)), y_one_hot)

        self.beta = beta

    def predict(self, x_test):
        """
        Predict label probabilities of new data using calculated beta.
        :param x_test: features of new data
        :return: class probabilities of new data
        """

        if self.kernel == 'rbf':
            kernel_func = pairwise.rbf_kernel(x_test, self.x_train)
        elif self.kernel == 'poly':
            kernel_func = pairwise.polynomial_kernel(x_test, self.x_train)
        elif self.kernel == 'sigmoid':
            kernel_func = pairwise.sigmoid_kernel(x_test, self.x_train)
        else:
            kernel_func = pairwise.linear_kernel(x_test, self.x_train)
        pred = np.matmul(kernel_func, self.beta)
        return pred
