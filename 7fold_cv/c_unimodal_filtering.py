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

csv_file = '7fold_cv_unimodal_filter.csv'
if not os.path.exists(csv_file):
    # If the file doesn't exist, create it and write the header
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)

"""
First the files with session-level functionals of the most emotional sentences 
need to be created
"""

names = ['top0.15', 'top0.2', 'top0.25', 'top0.3', 'top0.35', 'top0.4', 'top0.45', 'top0.5', 'top0.55', 'top0.6',
         'top0.65', 'top0.7', 'top0.75', 'top0.8']
tops = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]

for k in k_list:
    for vad_value in ['valence', 'arousal', 'dominance']:
        for i, top in enumerate(tops):

            loc_train = ''  # Loc of training file session-level embeddings for topX sentences of emotional dimension
            loc_dev = ''  # Loc of dev file session-level embeddings for topX sentences of emotional dimension
            X_tr_W2V = pd.read_csv(loc_train)
            X_de_W2V = pd.read_csv(loc_dev)
            X_W2V = pd.concat((X_tr_W2V, X_de_W2V))

            for f_list in list_combos_functionals:
                X_sub_W2V = X_W2V.iloc[:, X_W2V.columns.get_level_values(1).isin(f_list)]

                for c in c_list:
                    id_model_W2V = ['_'.join(f_list), c, k, vad_value, names[i]]

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

                        X_train_W2V_fold, X_test_W2V_fold = (X_sub_W2V.iloc[train_index, :],
                                                             X_sub_W2V.iloc[test_index, :])

                        _, _, pred_sev_W2V = sym_model(kelm_c=c, kelm_kernel=k, x_train=X_train_W2V_fold,
                                                       y_train=y_train_fold, x_input=X_test_W2V_fold)

                        pred_sev_list_W2V.append(pred_sev_W2V)

                        fold_df_W2V = pd.DataFrame({'pred_severity': pred_sev_W2V,
                                                    'true_severity': sev_test_fold.values,
                                                    'true_binary': bin_test_fold.values,
                                                    'gender': gen_test_fold.values})

                        perf_list_per_fold_W2V.append(root_mean_squared_error(sev_test_fold.values, pred_sev_W2V))

                    overall_pred_sev_W2V = list(itertools.chain.from_iterable(pred_sev_list_W2V))
                    new_row_overall_W2V = id_model_W2V + [root_mean_squared_error(overall_severity_vec,
                                                                                  overall_pred_sev_W2V)]
                    # Define the data for the new rows
                    with open(csv_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(new_row_overall_W2V)
