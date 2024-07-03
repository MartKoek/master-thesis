import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error
from evaluation.utilities import get_df_labels, get_functional_df_train_dev_test, return_7folds
from models import sym_model
import itertools
import csv
import os
import random

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

csv_file = '7fold_cv_decision_fusion.csv'
if not os.path.exists(csv_file):
    # If the file doesn't exist, create it and write the header
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)

# Select functionals based on earlier explorations
functs_SBERT = ['median']
functs_W2V = ['mean', 'var', 'median', 'max']
list_of_funct_lists = [functs_SBERT, functs_W2V]
c_list = [3, 2]
k_list = ["rbf", "linear", "poly", "sigmoid"]

""" Information about 7-fold cross validation """
random.seed(1)
folds = return_7folds(labels_total)  # list of 7 lists with equal division of classes
flat_indices = list(itertools.chain.from_iterable(folds))  # the order of indices in flattened folds
overall_binary_vec = binary_vec_total.iloc[flat_indices]
overall_severity_vec = severity_vec_total.iloc[flat_indices]
overall_gender_vec = gender_vec_total.iloc[flat_indices]

# Get all subsets
subset_X_train_SBERT = X_tr_SBERT.iloc[:, X_tr_SBERT.columns.get_level_values(1).isin(functs_SBERT)]
subset_X_train_W2V = X_tr_W2V.iloc[:, X_tr_W2V.columns.get_level_values(1).isin(functs_W2V)]
subset_X_dev_SBERT = X_de_SBERT.iloc[:, X_de_SBERT.columns.get_level_values(1).isin(functs_SBERT)]
subset_X_dev_W2V = X_de_W2V.iloc[:, X_de_W2V.columns.get_level_values(1).isin(functs_W2V)]

X_total_SBERT = pd.concat((subset_X_train_SBERT, subset_X_dev_SBERT))
X_total_W2V = pd.concat((subset_X_train_W2V, subset_X_dev_W2V))

# Explore all the different weightings of the audio and text modality
# The overall RMSE is saved, as well as the RMSE for males and females, allowing for gender-specific
# audio and text weighting

w_list = np.arange(0.0, 1, 0.01)

for k in k_list:
    for w_audio in w_list:

        pred_sev_list, performance_scores = [], []
        # Iterate through each fold
        for test_i in range(7):

            folds_use = folds.copy()
            test_index = folds_use.pop(test_i)  # take this list as test index set
            train_index = list(itertools.chain.from_iterable(folds_use))  # use the others as train indexes

            X_train_fold_SBERT, X_test_fold_SBERT = (X_total_SBERT.iloc[train_index, :],
                                                     X_total_SBERT.iloc[test_index, :])
            X_train_fold_W2V, X_test_fold_W2V = X_total_W2V.iloc[train_index, :], X_total_W2V.iloc[test_index, :]

            y_train_fold = symptoms_vec_total.iloc[train_index]

            gen_test_fold, sev_test_fold = gender_vec_total.iloc[test_index], severity_vec_total.iloc[test_index]
            bin_test_fold = binary_vec_total.iloc[test_index]

            _, _, pred_sev_sbert = sym_model(kelm_c=c_list[0], kelm_kernel=k, x_train=X_train_fold_SBERT,
                                             y_train=y_train_fold, x_input=X_test_fold_SBERT)
            _, _, pred_sev_w2v = sym_model(kelm_c=c_list[1], kelm_kernel=k, x_train=X_train_fold_W2V,
                                           y_train=y_train_fold, x_input=X_test_fold_W2V)

            w_text = 1 - w_audio
            pred_sev = (w_audio * pred_sev_w2v + w_text * pred_sev_sbert).astype(int)
            pred_sev_list.append(pred_sev)

        pred_severities = list(itertools.chain.from_iterable(pred_sev_list))
        total = pd.DataFrame({'pred_severity': pred_severities, 'true_severity': overall_severity_vec,
                              'true_binary': overall_binary_vec, 'gender': overall_gender_vec})
        g0_total = total.query('gender == 0')
        g1_total = total.query('gender == 1')

        new_row_overall = ['', w_audio] + [root_mean_squared_error(overall_severity_vec, pred_severities),
                                           root_mean_squared_error(g0_total['true_severity'],
                                                                   g0_total['pred_severity']),
                                           root_mean_squared_error(g1_total['true_severity'],
                                                                   g1_total['pred_severity'])]

        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(new_row_overall)
