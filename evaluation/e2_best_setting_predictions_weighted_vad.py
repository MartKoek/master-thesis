import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from constants import FUNCTIONALS, SBERT_MODELS
from evaluation.a_performance_metrics import performance_measures, binary_metrics
from evaluation.utilities import plot_pred_trues, get_df_labels, save_results_per_ckf
from models import sym_model, sev_model, bin_model
from scipy import stats

str_functs = '_'.join(FUNCTIONALS)

# Find labels of training, development, test set
labels_train, labels_dev, labels_test = get_df_labels()
train_gender_vec = labels_train['Gender']

train_symptoms = labels_train.drop(columns=['PHQ8_Binary', 'PHQ8_Score', 'Gender']).to_numpy()
dev_symptoms = labels_dev.drop(columns=['PHQ8_Binary', 'PHQ8_Score', 'Gender']).to_numpy()

train_binary_vec = labels_train['PHQ8_Binary']
dev_binary_vec = labels_dev['PHQ8_Binary']
test_binary_vec = labels_test['PHQ_Binary']

train_sev_vec = labels_train['PHQ8_Score']
dev_severity_vec = labels_dev['PHQ8_Score']
test_severity_vec = labels_test['PHQ_Score']


pre_list = ['sbert', 'w2v']
pre = pre_list[1]  # choose index
test_on = 'test'  # or 'test


""" Show performance with best settings based on lowest RMSE """

for vad_score in ['arousal', 'valence', 'dominance']:
    for str_abs in ['', '_abs']:
        if pre == 'w2v':
            setting_performances_df = pd.read_csv(f'C:/Users/mjkoe/Thesis/results/weighted_with_vad/w2v/'
                                                  f'{vad_score}{str_abs}_results.csv')
        else:
            setting_performances_df = pd.read_csv(f'C:/Users/mjkoe/Thesis/results/weighted_with_vad/sbert/'
                                                  f'{vad_score}{str_abs}_results.csv')

        best = setting_performances_df.query('RMSE == RMSE.min()').iloc[0]
        c = best['kelm_c']
        k = best['kelm_kernel']
        model = best['model']
        str_best_functs = best['functionals']
        if '_' in str_best_functs:
            f_list = str_best_functs.split('_')
        else:
            f_list = [str_best_functs]

        print(vad_score, str_abs)
        print(f_list, c)

        # Select the data to get results from
        X_train = pd.read_csv(f'C:/Users/mjkoe/Thesis/data/features/{model}/with_{vad_score}{str_abs}_train_functs.csv',
                              header=[0, 1], skipinitialspace=True, index_col=0)
        X_dev = pd.read_csv(f'C:/Users/mjkoe/Thesis/data/features/{model}/with_{vad_score}{str_abs}_dev_functs.csv',
                            header=[0, 1], skipinitialspace=True, index_col=0)
        X_test = pd.read_csv(f'C:/Users/mjkoe/Thesis/data/features/{model}/with_{vad_score}{str_abs}_test_functs.csv',
                             header=[0, 1], skipinitialspace=True, index_col=0)

        X_train = X_train.iloc[:, X_train.columns.get_level_values(1).isin(f_list)]
        X_dev = X_dev.iloc[:, X_dev.columns.get_level_values(1).isin(f_list)]
        X_test = X_test.iloc[:, X_test.columns.get_level_values(1).isin(f_list)]

        # Select the appropriate sets for symptom model
        if test_on == 'dev':
            X_input = X_dev
            y_train = train_symptoms
            input_severity_vec = dev_severity_vec
            input_binary_vec = dev_binary_vec
            gender_vec = labels_dev['Gender']
            input_index_vec = labels_dev
        else:
            y_train = np.concatenate((train_symptoms, dev_symptoms))
            X_train = pd.concat((X_train, X_dev))
            X_input = X_test
            input_severity_vec = test_severity_vec
            input_binary_vec = test_binary_vec
            gender_vec = labels_test['Gender']
            input_index_vec = labels_test


        """ The symptom model gives the best result """
        _, _, pred_sev = sym_model(kelm_c=c, kelm_kernel=k, x_train=X_train, y_train=y_train, x_input=X_input,
                                   classify=False, weight_kelm=False)
        df_symptom_results = pd.DataFrame({'pred_severity': pred_sev, 'true_severity': input_severity_vec,
                                           'true_binary': input_binary_vec, 'gender': gender_vec})
        performance_measures(df_symptom_results)

        # if vad_score == 'valence':
        #     plot_pred_trues(df_symptom_results)
        df_symptom_results.to_csv(
            path_or_buf=f'C:/Users/mjkoe/Thesis/results/predictions_best_settings/'
                        f'vad_weighted/{model}_symp_{pre}_{vad_score}{str_abs}_{test_on}.csv')
