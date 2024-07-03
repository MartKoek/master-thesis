import random
import pandas as pd
import numpy as np
from constants import FUNCTIONALS, SBERT_MODELS
from evaluation.a_performance_metrics import rmse_ratio_metrics, three_bin_f_metrics, print_perf_measures
from evaluation.utilities import get_df_labels
from models import sym_model
str_functs = '_'.join(FUNCTIONALS)

# Find labels of training, development, test set
labels_train, labels_dev, labels_test = get_df_labels()
labels_train_dev = pd.concat((labels_train, labels_dev))
y_train_symptoms = labels_train_dev.drop(columns=['PHQ8_Binary', 'PHQ8_Score', 'Gender'])

"""
Option for resampling bias mitigation method
"""
resample = False
def resample_group(group):
    if len(group) < max_count:
        return group.sample(max_count, replace=True)
    else:
        return group

if resample:
    nr_samples_score = labels_train_dev.groupby(['Gender', 'PHQ8_Score']).size()
    max_count = max(nr_samples_score)
    nr_samples_binary = labels_train_dev.groupby(['Gender', 'PHQ8_Binary']).size()
    max_count = max(nr_samples_binary)

    random.seed = 3

    # Apply the resampling function to each group
    resampled_df = labels_train_dev.groupby(['Gender', 'PHQ8_Score']).apply(resample_group)
    resampled_participant_vec = resampled_df.index.get_level_values(2)
    y_train_symptoms_resampled = y_train_symptoms.loc[resampled_participant_vec]

sbert = True  # Choose modality
if sbert:
    f_list = ['median']
    c = 3
else:
    f_list = ['mean', 'var', 'median', 'max']
    c = 2

k = 'linear'

""" Print performance with best settings based on lowest RMSE """

# Select the data to get results from
loc_train = ''  # Location of session-level embeddings for training set
loc_dev = ''  # Location of session-level embeddings for dev set
loc_test = ''  # Location of session-level embeddings for test set

X_train = pd.read_csv(loc_train)
X_dev = pd.read_csv(loc_dev)
X_test = pd.read_csv(loc_test)

X_combi = pd.concat((X_train, X_dev))
X_combined = X_combi.iloc[:, X_combi.columns.get_level_values(1).isin(f_list)]
X_test = X_test.iloc[:, X_test.columns.get_level_values(1).isin(f_list)]

""" The symptom model gives the best result """
_, _, pred_sev = sym_model(kelm_c=c, kelm_kernel=k, x_train=X_combined,
                           y_train=y_train_symptoms, x_input=X_test)
df_symptom_results = pd.DataFrame({'pred_severity': pred_sev, 'true_severity': labels_test['PHQ_Score'].values,
                                   'true_binary': labels_test['PHQ_Binary'].values,
                                   'gender': labels_test['Gender'].values})

print_perf_measures(df_symptom_results)