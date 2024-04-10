import pandas as pd
import numpy as np
import itertools
import pickle
from scipy import stats
import matplotlib.pyplot as plt
from constants import FUNCTIONALS
from utilities import get_df_labels, get_functional_df_train_dev_test
from preprocessing.utils import generate_functional_df
from models import sym_model
from evaluation.a_performance_metrics import performance_measures, non_binary_metrics

string_functionals = '_'.join(FUNCTIONALS)
labels_train, labels_dev, labels_test = get_df_labels()
train_symptoms = labels_train.drop(columns=['PHQ8_Binary', 'PHQ8_Score', 'Gender']).to_numpy()
dev_symptoms = labels_dev.drop(columns=['PHQ8_Binary', 'PHQ8_Score', 'Gender']).to_numpy()

dev_binary_vec = labels_dev['PHQ8_Binary']
test_binary_vec = labels_test['PHQ_Binary']

dev_severity_vec = labels_dev['PHQ8_Score']
test_severity_vec = labels_test['PHQ_Score']

(X_train_SBERT, X_dev_SBERT, X_test_SBERT, X_train_W2V, X_dev_W2V, X_test_W2V,
 X_train_VAD, X_dev_VAD, X_test_VAD) = get_functional_df_train_dev_test()

test_on = 'dev'

# Select the appropriate sets for symptom model
if test_on == 'dev':
    y_train = train_symptoms
    input_severity_vec = dev_severity_vec
    input_binary_vec = dev_binary_vec
    gender_vec = labels_dev['Gender']
    input_index_vec = labels_dev
else:
    y_train = np.concatenate((train_symptoms, dev_symptoms))
    input_severity_vec = test_severity_vec
    input_binary_vec = test_binary_vec
    gender_vec = labels_test['Gender']
    input_index_vec = labels_test


# Select functionals based on earlier explorations
functs_SBERT = ['mean', 'median'] # or mean_var_median
functs_W2V = ['var', 'std', 'quantile'] # var_std_median_quantile_max
functs_VAD = ["mean", "max"]

# Get all subsets
subset_X_train_SBERT = X_train_SBERT.iloc[:, X_train_SBERT.columns.get_level_values(1).isin(functs_SBERT)]
subset_X_train_W2V = X_train_W2V.iloc[:, X_train_W2V.columns.get_level_values(1).isin(functs_W2V)]
subset_X_train_VAD = X_train_VAD.iloc[:, X_train_VAD.columns.get_level_values(1).isin(functs_VAD)]

subset_X_dev_SBERT = X_dev_SBERT.iloc[:, X_dev_SBERT.columns.get_level_values(1).isin(functs_SBERT)]
subset_X_dev_W2V = X_dev_W2V.iloc[:, X_dev_W2V.columns.get_level_values(1).isin(functs_W2V)]
subset_X_dev_VAD = X_dev_VAD.iloc[:, X_dev_VAD.columns.get_level_values(1).isin(functs_VAD)]

subset_X_test_SBERT = X_test_SBERT.iloc[:, X_test_SBERT.columns.get_level_values(1).isin(functs_SBERT)]
subset_X_test_W2V = X_test_W2V.iloc[:, X_test_W2V.columns.get_level_values(1).isin(functs_W2V)]
subset_X_test_VAD = X_test_VAD.iloc[:, X_test_VAD.columns.get_level_values(1).isin(functs_VAD)]



c_list = range(2,100)

""" 1. SBERT & PDEM """

print('\nFuse SBERT & PDEM')
subset_X_train_SBERT_W2V = pd.concat([subset_X_train_SBERT, subset_X_train_W2V], axis=1)
subset_X_dev_SBERT_W2V = pd.concat([subset_X_dev_SBERT, subset_X_dev_W2V], axis=1)
subset_X_test_SBERT_W2V = pd.concat([subset_X_test_SBERT, subset_X_test_W2V], axis=1)

if test_on == 'dev':
    X_input = subset_X_dev_SBERT_W2V
    X_train = subset_X_train_SBERT_W2V
else:
    X_input = subset_X_test_SBERT_W2V
    X_train = pd.concat((subset_X_train_SBERT_W2V, subset_X_dev_SBERT_W2V))
# c_list = [3]

results_sbert_W2V = []
for c in c_list:
    _, _, pred_sev = sym_model(kelm_c=c, kelm_kernel='linear', x_train=X_train, y_train=y_train, x_input=X_input)
    results_sbert_W2V.append(non_binary_metrics(trues=input_severity_vec, preds=pred_sev)[0])

print(min(results_sbert_W2V), c_list[np.argmin(results_sbert_W2V)])
if test_on == 'dev':
    plt.plot(c_list, results_sbert_W2V, label='SBERT & PDEM')

df_pred_trues = pd.DataFrame({'pred_severity': pred_sev, 'true_severity': input_severity_vec,
                              'true_binary': input_binary_vec, 'gender': gender_vec})
performance_measures(df_pred_trues)


""" 2. BERT & VAD """

print('\nFuse SBERT & VAD')
results_sbert_vad = []
subset_X_train_SBERT_VAD = pd.concat([subset_X_train_SBERT, subset_X_train_VAD], axis=1)
subset_X_dev_SBERT_VAD = pd.concat([subset_X_dev_SBERT, subset_X_dev_VAD], axis=1)
subset_X_test_SBERT_VAD = pd.concat([subset_X_test_SBERT, subset_X_test_VAD], axis=1)

if test_on == 'dev':
    X_input = subset_X_dev_SBERT_VAD
    X_train = subset_X_train_SBERT_VAD
else:
    X_input = subset_X_test_SBERT_VAD
    X_train = pd.concat((subset_X_train_SBERT_VAD, subset_X_dev_SBERT_VAD))
# c_list = [52]

for c in c_list:
    _, _, pred_sev = sym_model(kelm_c=c, kelm_kernel='linear', x_train=X_train, y_train=y_train, x_input=X_input)
    results_sbert_vad.append(non_binary_metrics(trues=input_severity_vec, preds=pred_sev)[0])

print(min(results_sbert_vad), c_list[np.argmin(results_sbert_vad)])
if test_on == 'dev':
    plt.plot(c_list, results_sbert_vad, label='SBERT & VAD')

df_pred_trues = pd.DataFrame({'pred_severity': pred_sev, 'true_severity': input_severity_vec,
                              'true_binary': input_binary_vec, 'gender': gender_vec})
performance_measures(df_pred_trues)


""" 3. VAD & PDEM """

print('\nFuse PDEM & VAD')
results_pdem_vad = []
subset_X_train_W2V_VAD = pd.concat([subset_X_train_W2V, subset_X_train_VAD], axis=1)
subset_X_dev_W2V_VAD = pd.concat([subset_X_dev_W2V, subset_X_dev_VAD], axis=1)
subset_X_test_W2V_VAD = pd.concat([subset_X_test_W2V, subset_X_test_VAD], axis=1)

if test_on == 'dev':
    X_input = subset_X_dev_W2V_VAD
    X_train = subset_X_train_W2V_VAD
else:
    X_input = subset_X_test_W2V_VAD
    X_train = pd.concat((subset_X_train_W2V_VAD, subset_X_dev_W2V_VAD))
# c_list = [32]

for c in c_list:
    _, _, pred_sev = sym_model(kelm_c=c, kelm_kernel='linear', x_train=X_train, y_train=y_train, x_input=X_input)
    results_pdem_vad.append(non_binary_metrics(trues=input_severity_vec, preds=pred_sev)[0])

print(min(results_pdem_vad), c_list[np.argmin(results_pdem_vad)])
if test_on == 'dev':
    plt.plot(c_list, results_pdem_vad, label = 'PDEM & VAD')

df_pred_trues = pd.DataFrame({'pred_severity': pred_sev, 'true_severity': input_severity_vec,
                              'true_binary': input_binary_vec, 'gender': gender_vec})
performance_measures(df_pred_trues)


""" 4. BERT & PDEM & VAD """

results_all = []
print('\nFuse SBERT & PDEM & VAD')
subset_X_train_SBERT_W2V_VAD = pd.concat([subset_X_train_SBERT, subset_X_train_W2V, subset_X_train_VAD], axis=1)
subset_X_dev_SBERT_W2V_VAD = pd.concat([subset_X_dev_SBERT, subset_X_dev_W2V, subset_X_dev_VAD], axis=1)
subset_X_test_SBERT_W2V_VAD = pd.concat([subset_X_test_SBERT, subset_X_test_W2V, subset_X_test_VAD], axis=1)

if test_on == 'dev':
    X_input = subset_X_dev_SBERT_W2V_VAD
    X_train = subset_X_train_SBERT_W2V_VAD
else:
    X_input = subset_X_test_SBERT_W2V_VAD
    X_train = pd.concat((subset_X_train_SBERT_W2V_VAD, subset_X_dev_SBERT_W2V_VAD))
# c_list = [3]

for c in c_list:
    _, _, pred_sev = sym_model(kelm_c=c, kelm_kernel='linear', x_train=X_train, y_train=y_train, x_input=X_input)
    results_all.append(non_binary_metrics(trues=input_severity_vec, preds=pred_sev)[0])

print(min(results_all), c_list[np.argmin(results_all)])
if test_on == 'dev':
    plt.plot(c_list, results_all, label = 'SBERT & PDEM & VAD')

df_pred_trues = pd.DataFrame({'pred_severity': pred_sev, 'true_severity': input_severity_vec,
                          'true_binary': input_binary_vec, 'gender': gender_vec})
performance_measures(df_pred_trues)

if test_on == 'dev':
    plt.legend()
    plt.ylabel('RMSE')
    plt.xlabel('C in Kernel ELM')
    plt.title('Performance feature fusion dev set')
    plt.show()