import pandas as pd
from sklearn.metrics import root_mean_squared_error
from constants import FUNCTIONALS
from evaluation.utilities import get_df_labels, return_7folds
from models import sym_model
import itertools
import csv
import os
import random

""" 
Use this file to test all different settings for the attention models
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

csv_file = '7fold_cv_unimodal_attention.csv'
if not os.path.exists(csv_file):
    # If the file doesn't exist, create it and write the header
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)

sbert_or_pdem = 'sbert_miniLM_weighted'
# sbert_or_pdem = 'pdem_weighted'

for vad_value in ['arousal', 'valence', 'dominance']:
    for abs_value in ['', '_abs']:
        if sbert_or_pdem != 'pdem_weighted':
            loc_train_SBERT = ''  # Add the location to the file with session-level SBERT embeddings for training set
            loc_dev_SBERT = ''  # Add the location to the file with session-level SBERT embeddings for training set
            X_tr_SBERT = pd.read_csv(loc_train_SBERT)
            X_de_SBERT = pd.read_csv(loc_dev_SBERT)
            X_SBERT = pd.concat((X_tr_SBERT, X_de_SBERT))

        else:
            loc_train_W2V = ''  # Add the location to the file with session-level PDEM embeddings for training set
            loc_dev_W2V = ''  # Add the location to the file with session-level PDEM embeddings for training set
            X_tr_W2V = pd.read_csv(loc_train_W2V)
            X_de_W2V = pd.read_csv(loc_dev_W2V)
            X_W2V = pd.concat((X_tr_W2V, X_de_W2V))

        # Loop over all different combinations of functionals
        for f_list in list_combos_functionals:
            if sbert_or_pdem != 'pdem_weighted':
                X_sub_SBERT = X_SBERT.iloc[:, X_SBERT.columns.get_level_values(1).isin(f_list)]
            else:
                X_sub_W2V = X_W2V.iloc[:, X_W2V.columns.get_level_values(1).isin(f_list)]

            for k in ['linear']:
                for c in c_list:
                    id_model_SBERT = [sbert_or_pdem, '_'.join(f_list), c, k, vad_value, abs_value]
                    pred_sev_list_SBERT, perf_list_per_fold_SBERT = [], []

                    id_model_W2V = [sbert_or_pdem, '_'.join(f_list), c, k, vad_value, abs_value]
                    pred_sev_list_W2V, perf_list_per_fold_W2V = [], []

                    # Iterate through each fold
                    for test_i in range(7):
                        folds_use = folds.copy()
                        test_index = folds_use.pop(test_i)  # take this list as test index set
                        train_index = list(itertools.chain.from_iterable(folds_use))  # use the others as train indexes

                        y_train_fold = symptoms_vec_total.iloc[train_index]
                        gen_test_fold, sev_test_fold = (gender_vec_total.iloc[test_index],
                                                        severity_vec_total.iloc[test_index])
                        bin_test_fold = binary_vec_total.iloc[test_index]

                        if sbert_or_pdem != 'pdem_weighted':
                            X_train_SBERT_fold, X_test_SBERT_fold = (X_sub_SBERT.iloc[train_index, :],
                                                                     X_sub_SBERT.iloc[test_index, :])
                            _, _, pred_sev_SBERT = sym_model(kelm_c=c, kelm_kernel=k, x_train=X_train_SBERT_fold,
                                                             y_train=y_train_fold,
                                                             x_input=X_test_SBERT_fold)
                            pred_sev_list_SBERT.append(pred_sev_SBERT)

                        else:
                            X_train_W2V_fold, X_test_W2V_fold = (X_sub_W2V.iloc[train_index, :],
                                                                 X_sub_W2V.iloc[test_index, :])
                            _, _, pred_sev_W2V = sym_model(kelm_c=c, kelm_kernel=k, x_train=X_train_W2V_fold,
                                                           y_train=y_train_fold,
                                                           x_input=X_test_W2V_fold)
                            pred_sev_list_W2V.append(pred_sev_W2V)

                    if sbert_or_pdem != 'pdem_weighted':
                        overall_pred_sev_SBERT = list(itertools.chain.from_iterable(pred_sev_list_SBERT))
                        new_row_overall_SBERT = id_model_SBERT + [root_mean_squared_error(overall_severity_vec,
                                                                                          overall_pred_sev_SBERT)]
                    else:
                        overall_pred_sev_W2V = list(itertools.chain.from_iterable(pred_sev_list_W2V))
                        new_row_overall_W2V = id_model_W2V + [root_mean_squared_error(overall_severity_vec,
                                                                                      overall_pred_sev_W2V)]
                    with open(csv_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        if sbert_or_pdem != 'pdem_weighted':
                            writer.writerow(new_row_overall_SBERT)
                        else:
                            writer.writerow(new_row_overall_W2V)
