import pandas as pd
from constants import FUNCTIONALS
from evaluation.a_performance_metrics import print_perf_measures, rmse_ratio_metrics, three_bin_f_metrics
from evaluation.utilities import get_df_labels
from models import sym_model
str_functs = '_'.join(FUNCTIONALS)

# Find labels of training, development, test set
labels_train, labels_dev, labels_test = get_df_labels()
labels_train_dev = pd.concat((labels_train, labels_dev))
y_train_symptoms = labels_train_dev.drop(columns=['PHQ8_Binary', 'PHQ8_Score', 'Gender'])

k = 'linear'

# Select modality and parameters per modality
sbert = False
if sbert:
    f_list = ['quantile']
    c = 10
    vad = 'arousal'
    abs_value = ''  # or _abs

else:
    f_list = ['mean', 'var', 'std', 'median', 'max']
    c = 9
    vad = 'arousal'
    abs_value = ''  # or _abs

""" Show performance with best settings based on lowest RMSE """

# Select the data to get results from
loc_train = ''  # Location of train file with session-level embeddings (correct model, vad value, abs_value value)
loc_dev = ''  # Location of dev file with session-level embeddings (correct model, vad value, abs_value value)
loc_test = ''  # Location of dev file with session-level embeddings (correct model, vad value, abs_value value)

X_train = pd.read_csv(loc_train)
X_dev = pd.read_csv(loc_dev)
X_test = pd.read_csv(loc_test)

X_combi = pd.concat((X_train, X_dev))
X_combined = X_combi.iloc[:, X_combi.columns.get_level_values(1).isin(f_list)]  # Select best functionals
X_test = X_test.iloc[:, X_test.columns.get_level_values(1).isin(f_list)]

""" The symptom model gives the best result """
_, _, pred_sev = sym_model(kelm_c=c, kelm_kernel=k, x_train=X_combined,
                           y_train=y_train_symptoms, x_input=X_test)
df_symptom_results = pd.DataFrame({'pred_severity': pred_sev, 'true_severity': labels_test['PHQ_Score'].values,
                                   'true_binary': labels_test['PHQ_Binary'].values,
                                   'gender': labels_test['Gender'].values})

print_perf_measures(df_symptom_results)  # Print performance metrics

print(rmse_ratio_metrics(df_symptom_results)[-1])  # Equal accuracy
print(three_bin_f_metrics(df_symptom_results)[2])  # Equal opportunity
print(three_bin_f_metrics(df_symptom_results)[5])  # Predictive equality
