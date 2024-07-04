import pandas as pd
from constants import FUNCTIONALS
from evaluation.utilities import get_df_labels, get_functional_df_train_dev_test
from models import sym_model
from evaluation.a_performance_metrics import print_perf_measures

string_functionals = '_'.join(FUNCTIONALS)

# Find labels of training, development, test set
labels_train, labels_dev, labels_test = get_df_labels()
labels_train_dev = pd.concat((labels_train, labels_dev))
y_train_symptoms = labels_train_dev.drop(columns=['PHQ8_Binary', 'PHQ8_Score', 'Gender'])

(X_train_SBERT, X_dev_SBERT, X_test_SBERT, X_train_W2V, X_dev_W2V, X_test_W2V,
 X_train_VAD, X_dev_VAD, X_test_VAD) = get_functional_df_train_dev_test()

X_combi_SBERT = pd.concat((X_train_SBERT, X_dev_SBERT))
X_combi_W2V = pd.concat((X_train_W2V, X_dev_W2V))

# Select functionals based on earlier explorations
f_lists = [['median'], ['mean', 'var', 'median', 'max']]
c_list = [3, 2]
k = 'linear'

# Get all subsets
subset_X_combi_SBERT = X_combi_SBERT.iloc[:, X_combi_SBERT.columns.get_level_values(1).isin(f_lists[0])]
subset_X_combi_W2V = X_combi_W2V.iloc[:, X_combi_W2V.columns.get_level_values(1).isin(f_lists[1])]

subset_X_test_SBERT = X_test_SBERT.iloc[:, X_test_SBERT.columns.get_level_values(1).isin(f_lists[0])]
subset_X_test_W2V = X_test_W2V.iloc[:, X_test_W2V.columns.get_level_values(1).isin(f_lists[1])]

""" The symptom model gives the best result """
_, _, pred_sev_SBERT = sym_model(kelm_c=c_list[0], kelm_kernel=k, x_train=subset_X_combi_SBERT,
                                 y_train=y_train_symptoms, x_input=subset_X_test_SBERT)
_, _, pred_sev_W2V = sym_model(kelm_c=c_list[1], kelm_kernel=k, x_train=subset_X_combi_W2V,
                               y_train=y_train_symptoms, x_input=subset_X_test_W2V)


"""
Equal weighting (DF)
"""
w_audio = 0.5
w_text = 1 - w_audio
pred_sev = (w_audio * pred_sev_W2V + w_text * pred_sev_SBERT).astype(int)
df_bimodal_results = pd.DataFrame({'pred_severity': pred_sev, 'true_severity': labels_test['PHQ_Score'].values,
                                   'true_binary': labels_test['PHQ_Binary'].values,
                                   'gender': labels_test['Gender'].values})
print_perf_measures(df_bimodal_results)

"""
Optimal weighting based on validation set (WDF)
"""
w_audio = 0.2
w_text = 1
pred_sev = (w_audio * pred_sev_W2V + w_text * pred_sev_SBERT).astype(int)
df_bimodal_results = pd.DataFrame({'pred_severity': pred_sev, 'true_severity': labels_test['PHQ_Score'].values,
                                   'true_binary': labels_test['PHQ_Binary'].values,
                                   'gender': labels_test['Gender'].values})
print_perf_measures(df_bimodal_results)

"""
Gender-optimal weighting (GWDF)
"""
w_audio_g0 = 0.2
w_audio_g1 = 0.51
w_text_g0 = 1 - w_audio_g0
w_text_g1 = 1 - w_audio_g1

pred_sev_g0 = (w_audio_g0 * pred_sev_W2V + w_text_g0 * pred_sev_SBERT).astype(int)
pred_sev_g1 = (w_audio_g1 * pred_sev_W2V + w_text_g1 * pred_sev_SBERT).astype(int)

df_g0 = pd.DataFrame({'pred_severity': pred_sev_g0, 'true_severity': labels_test['PHQ_Score'].values,
                      'true_binary': labels_test['PHQ_Binary'].values,
                      'gender': labels_test['Gender'].values})
df_g1 = pd.DataFrame({'pred_severity': pred_sev_g1, 'true_severity': labels_test['PHQ_Score'].values,
                      'true_binary': labels_test['PHQ_Binary'].values,
                      'gender': labels_test['Gender'].values})

df_g0_new = df_g0.query('gender == 0')
df_g1_new = df_g1.query('gender == 1')
together = pd.concat([df_g0_new, df_g1_new])

print_perf_measures(together)
