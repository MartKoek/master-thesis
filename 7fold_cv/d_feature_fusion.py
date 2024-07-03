import pandas as pd
import os
import csv
import itertools
from sklearn.metrics import root_mean_squared_error
from constants import FUNCTIONALS
from evaluation.utilities import get_df_labels, get_functional_df_train_dev_test, return_7folds
from models import sym_model
import random

string_functionals = '_'.join(FUNCTIONALS)

# Find labels of training, development, test set
labels_train, labels_dev, _ = get_df_labels()
labels_total = pd.concat((labels_train, labels_dev))
gender_vec_total = labels_total['Gender']
binary_vec_total = labels_total['PHQ8_Binary']
severity_vec_total = labels_total['PHQ8_Score']
symptoms_vec_total = labels_total.drop(columns=['PHQ8_Binary', 'PHQ8_Score', 'Gender'])

X_tr_SBERT, X_de_SBERT, _, X_tr_W2V, X_de_W2V, _, _, _, _ = get_functional_df_train_dev_test()

X_SBERT = pd.concat((X_tr_SBERT, X_de_SBERT))
X_W2V = pd.concat((X_tr_W2V, X_de_W2V))

""" Set ranges for parameters """
k_list = ["rbf", "linear", "poly", "sigmoid"]
c_list = range(1, 20)

""" Information about 7-fold cross validation """
random.seed(1)
folds = return_7folds(labels_total)  # list of 7 lists with equal division of classes
flat_indices = list(itertools.chain.from_iterable(folds))  # the order of indices in flattened folds
overall_binary_vec = binary_vec_total.iloc[flat_indices]
overall_severity_vec = severity_vec_total.iloc[flat_indices]
overall_gender_vec = gender_vec_total.iloc[flat_indices]

""" File to save results """

csv_file = '7fold_cv_feature_fusion.csv'
if not os.path.exists(csv_file):
    # If the file doesn't exist, create it and write the header
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)

# Select functionals based on earlier explorations
functs_SBERT = ['median']
functs_W2V = ['mean', 'var', 'median', 'max']

# Get all subsets
subset_X_train_SBERT = X_tr_SBERT.iloc[:, X_tr_SBERT.columns.get_level_values(1).isin(functs_SBERT)]
subset_X_train_W2V = X_tr_W2V.iloc[:, X_tr_W2V.columns.get_level_values(1).isin(functs_W2V)]
subset_X_dev_SBERT = X_de_SBERT.iloc[:, X_de_SBERT.columns.get_level_values(1).isin(functs_SBERT)]
subset_X_dev_W2V = X_de_W2V.iloc[:, X_de_W2V.columns.get_level_values(1).isin(functs_W2V)]

X_total_SBERT = pd.concat((subset_X_train_SBERT, subset_X_dev_SBERT))
X_total_W2V = pd.concat((subset_X_train_W2V, subset_X_dev_W2V))


""" Define feature fused vectors"""
X = pd.concat([X_total_SBERT, X_total_W2V], axis=1)

for k in k_list:
    for c in c_list:
        id_model = [c, k]

        pred_sev_list, performance_scores = [], []

        # Iterate through each fold
        for test_i in range(7):
            folds_use = folds.copy()
            test_index = folds_use.pop(test_i)  # take this list as test index set
            train_index = list(itertools.chain.from_iterable(folds_use))  # use the others as train indexes

            X_train_fold, X_test_fold = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train_fold = symptoms_vec_total.iloc[train_index]

            gen_test_fold, sev_test_fold = gender_vec_total.iloc[test_index], severity_vec_total.iloc[test_index]
            bin_test_fold = binary_vec_total.iloc[test_index]

            _, _, pred_sev = sym_model(kelm_c=c, kelm_kernel=k, x_train=X_train_fold, y_train=y_train_fold,
                                       x_input=X_test_fold)

            pred_sev_list.append(pred_sev)

        pred_severities = list(itertools.chain.from_iterable(pred_sev_list))
        new_row_overall = id_model + [root_mean_squared_error(overall_severity_vec, pred_severities)]

        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(new_row_overall)
