import pandas as pd
from sklearn.metrics import root_mean_squared_error
from constants import FUNCTIONALS
from evaluation.utilities import get_df_labels, return_7folds, get_functional_df_train_dev_test
from models import sym_model
import itertools
import csv
import os
import random

""" 
Use this file to test all different settings for the plain unimodal models
"""

# A list of all possible combinations of functionals
str_functs = '_'.join(FUNCTIONALS)
list_combos_functionals = []
for index in range(1, len(FUNCTIONALS)):
    for list_of_functionals in itertools.combinations(FUNCTIONALS, index):
        list_combos_functionals.append(list(list_of_functionals))

# Find labels of training, development, test set
labels_train, labels_dev, _ = get_df_labels()
labels_total = pd.concat((labels_train, labels_dev))
gender_vec_total = labels_total['Gender']
binary_vec_total = labels_total['PHQ8_Binary']
severity_vec_total = labels_total['PHQ8_Score']
symptoms_vec_total = labels_total.drop(columns=['PHQ8_Binary', 'PHQ8_Score', 'Gender'])

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

csv_file = '7fold_cv_unimodal.csv'
if not os.path.exists(csv_file):
    # If the file doesn't exist, create it and write the header
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)


X_tr_SBERT, X_de_SBERT, _, X_tr_W2V, X_de_W2V, _, _, _, _, = get_functional_df_train_dev_test()
X_SBERT = pd.concat((X_tr_SBERT, X_de_SBERT))
X_W2V = pd.concat((X_tr_W2V, X_de_W2V))

# Loop over all different combinations of functionals
for f_list in list_combos_functionals:
    X_sub_SBERT = X_SBERT.iloc[:, X_SBERT.columns.get_level_values(1).isin(f_list)]
    X_sub_W2V = X_W2V.iloc[:, X_W2V.columns.get_level_values(1).isin(f_list)]

    for k in k_list:
        for c in c_list:

            id_sbert = ['sbert', '_'.join(f_list), c, k]
            id_pdem = ['pdem', '_'.join(f_list), c, k]

            pred_sev_list_SBERT, perf_list_per_fold_SBERT = [], []
            pred_sev_list_W2V, perf_list_per_fold_W2V = [], []

            # Iterate through each fold
            for test_i in range(7):
                folds_use = folds.copy()
                test_index = folds_use.pop(test_i)  # take this list as test index set
                train_index = list(itertools.chain.from_iterable(folds_use))  # use the others as train indexes

                y_train_fold = symptoms_vec_total.iloc[train_index]
                gen_test_fold, sev_test_fold = gender_vec_total.iloc[test_index], severity_vec_total.iloc[test_index]
                bin_test_fold = binary_vec_total.iloc[test_index]

                X_train_SBERT_fold, X_test_SBERT_fold = (X_sub_SBERT.iloc[train_index, :],
                                                         X_sub_SBERT.iloc[test_index, :])
                X_train_W2V_fold, X_test_W2V_fold = (X_sub_W2V.iloc[train_index, :],
                                                     X_sub_W2V.iloc[test_index, :])

                _, _, pred_sev_SBERT = sym_model(kelm_c=c, kelm_kernel=k, x_train=X_train_SBERT_fold,
                                                 y_train=y_train_fold, x_input=X_test_SBERT_fold)
                _, _, pred_sev_W2V = sym_model(kelm_c=c, kelm_kernel=k, x_train=X_train_W2V_fold,
                                               y_train=y_train_fold, x_input=X_test_W2V_fold)

                pred_sev_list_SBERT.append(pred_sev_SBERT)
                pred_sev_list_W2V.append(pred_sev_W2V)

            # the predicted overall_severity_vec concatenated
            overall_pred_sev_SBERT = list(itertools.chain.from_iterable(pred_sev_list_SBERT))
            overall_pred_sev_W2V = list(itertools.chain.from_iterable(pred_sev_list_W2V))

            new_row_overall_SBERT = id_sbert + [root_mean_squared_error(overall_severity_vec, overall_pred_sev_SBERT)]
            new_row_overall_W2V = id_pdem + [root_mean_squared_error(overall_severity_vec, overall_pred_sev_W2V)]

            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(new_row_overall_SBERT)
                writer.writerow(new_row_overall_W2V)
