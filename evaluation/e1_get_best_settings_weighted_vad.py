import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from constants import FUNCTIONALS, SBERT_MODELS
from evaluation.a_performance_metrics import performance_measures, binary_metrics, non_binary_metrics
from evaluation.utilities import plot_pred_trues, get_df_labels, save_results_per_ckf
from models import sym_model, sev_model, bin_model
from scipy import stats
import itertools

str_functs = '_'.join(FUNCTIONALS)
list_combos_functionals = []
for index in range(1, len(FUNCTIONALS)):
    for list_of_functionals in itertools.combinations(FUNCTIONALS, index):
        list_combos_functionals.append(list(list_of_functionals))


# Find labels of training, development, test set
labels_train, labels_dev, labels_test = get_df_labels()
train_gender_vec = labels_train['Gender']

train_binary_vec = labels_train['PHQ8_Binary']
train_sev_vec = labels_train['PHQ8_Score']
train_symptoms = labels_train.drop(columns=['PHQ8_Binary', 'PHQ8_Score', 'Gender']).to_numpy()

dev_binary_vec = labels_dev['PHQ8_Binary']
dev_severity_vec = labels_dev['PHQ8_Score']
dev_symptoms = labels_dev.drop(columns=['PHQ8_Binary', 'PHQ8_Score', 'Gender']).to_numpy()

input_severity_vec = dev_severity_vec
input_binary_vec = dev_binary_vec
gender_vec = labels_dev['Gender']
input_index_vec = labels_dev

k = 'linear'

get_res_sbert = True
get_res_w2v = True

models = ['sbert_miniLM_weighted', 'pdem_weighted']
pres = ['sbert', 'w2v']
vad_score = ['valence', 'arousal', 'dominance']
str_absolute = ['', '_abs']


c_lists = [[50, 55, 57, 60, 63, 65, 68, 70, 73, 75],
           [50, 55, 57, 60, 63,  65, 68, 70, 73, 75]]


for i, modality in enumerate([get_res_sbert, get_res_w2v]):
    if modality:
        for score in vad_score:
            for str_abs in str_absolute:
                X_train = pd.read_csv(f'C:/Users/mjkoe/Thesis/data/features/{models[i]}/with_{score}{str_abs}_'
                                      f'train_functs.csv', header=[0, 1], skipinitialspace=True, index_col=0)
                X_dev = pd.read_csv(f'C:/Users/mjkoe/Thesis/data/features/{models[i]}/with_{score}{str_abs}_'
                                    f'dev_functs.csv', header=[0, 1], skipinitialspace=True, index_col=0)

                results = []
                for c in c_lists[i]:
                    for f_list in list_combos_functionals:
                        X_train_subset = X_train.iloc[:, X_train.columns.get_level_values(1).isin(f_list)]
                        X_input_subset = X_dev.iloc[:, X_dev.columns.get_level_values(1).isin(f_list)]

                        _, _, pred_sev1 = sym_model(kelm_c=c, kelm_kernel=k, x_train=X_train_subset, y_train=train_symptoms,
                                                    x_input=X_input_subset, classify=False, weight_kelm=False)
                        pred_sev2 = sev_model(kelm_c=c, kelm_kernel=k, x_train=X_train_subset, y_train=train_sev_vec,
                                              x_input=X_input_subset, classify=False, weight_kelm=False)

                        rmse, ccc, mae = non_binary_metrics(trues=input_severity_vec, preds=pred_sev1)
                        results.append(['symptom_model', models[i], '_'.join(f_list), c, k, rmse, ccc, mae])
                        rmse, ccc, mae = non_binary_metrics(trues=input_severity_vec, preds=pred_sev2)
                        results.append(['sev_model', models[i], '_'.join(f_list), c, k, rmse, ccc, mae])

                df_results = pd.DataFrame(results)
                df_results.columns = ['method', 'model', 'functionals', 'kelm_c', 'kelm_kernel', 'RMSE', 'CCC', 'MAE']
                df_results.to_csv(f'C:/Users/mjkoe/Thesis/results/weighted_with_vad/{pres[i]}/'
                                  f'{score}{str_abs}_results.csv')
