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

# Get all subsets
subset_X_combi_SBERT = X_combi_SBERT.iloc[:, X_combi_SBERT.columns.get_level_values(1).isin(f_lists[0])]
subset_X_combi_W2V = X_combi_W2V.iloc[:, X_combi_W2V.columns.get_level_values(1).isin(f_lists[1])]

subset_X_test_SBERT = X_test_SBERT.iloc[:, X_test_SBERT.columns.get_level_values(1).isin(f_lists[0])]
subset_X_test_W2V = X_test_W2V.iloc[:, X_test_W2V.columns.get_level_values(1).isin(f_lists[1])]

print('\nFuse SBERT & PDEM')
subset_X_train_SBERT_W2V = pd.concat([subset_X_combi_SBERT, subset_X_combi_W2V], axis=1)
subset_X_test_SBERT_W2V = pd.concat([subset_X_test_SBERT, subset_X_test_W2V], axis=1)

c = 2

""" 1. SBERT & PDEM """
_, _, pred_sev = sym_model(kelm_c=c, kelm_kernel='linear', x_train=subset_X_train_SBERT_W2V,
                           y_train=y_train_symptoms, x_input=subset_X_test_SBERT_W2V)
df_pred_trues = pd.DataFrame({'pred_severity': pred_sev, 'true_severity': labels_test['PHQ_Score'].values,
                              'true_binary': labels_test['PHQ_Binary'].values, 'gender': labels_test['Gender'].values})

print_perf_measures(df_pred_trues)
