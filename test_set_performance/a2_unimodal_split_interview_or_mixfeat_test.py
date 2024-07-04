import pandas as pd
from constants import FUNCTIONALS
from evaluation.a_performance_metrics import print_perf_measures
from evaluation.utilities import get_df_labels
from models import sym_model
str_functs = '_'.join(FUNCTIONALS)

# Find labels of training, development, test set
labels_train, labels_dev, labels_test = get_df_labels()
labels_train_dev = pd.concat((labels_train, labels_dev))
y_train_symptoms = labels_train_dev.drop(columns=['PHQ8_Binary', 'PHQ8_Score', 'Gender'])

sbert = True  # Choose modality
if sbert:
    f_list = ['median']
    c = 3
else:
    f_list = ['mean', 'var', 'median', 'max']
    c = 2

k = 'linear'

""" Show performance with best settings based on lowest RMSE """
# Select the data to get results from
loc_train_dev_file = ''  # Location file with original + generated session-level samples (split interview or mixfeat)
loc_test_file = ''  # Location of the test file with session embeddings

X_combi = pd.read_csv(loc_train_dev_file)
X_test = pd.read_csv(loc_test_file)

# Select the best functionals
X_combined = X_combi.iloc[:, X_combi.columns.get_level_values(1).isin(f_list)]
X_combined.index = X_combined.index.astype('str')

X_test = X_test.iloc[:, X_test.columns.get_level_values(1).isin(f_list)]

# Indexes of original samples and generated samples (e.g., 304 and 304A)
X_sub_original = X_combined[X_combined.index.str.len() == 3]
X_sub_original.index = X_sub_original.index.astype(int)
X_sub_generated = X_combined[X_combined.index.str.len() == 4].copy()
X_sub_generated['original_ID'] = X_sub_generated.index.str[:3].astype(int)

fold_train_info = pd.concat((labels_train_dev['Gender'], labels_train_dev['PHQ8_Binary']),
                            axis='columns')

# Distinguish between classes
list_depr_males = list(fold_train_info.query('PHQ8_Binary == 1 & Gender == 1').index)
list_depr_fem = list(fold_train_info.query('PHQ8_Binary == 1 & Gender == 0').index)
list_not_depr_fem = list(fold_train_info.query('PHQ8_Binary == 0 & Gender == 0').index)
list_not_depr_males = list(fold_train_info.query('PHQ8_Binary == 0 & Gender == 1').index)

extra_depr_males_needed = len(list_not_depr_males) - len(list_depr_males)
extra_depr_fem_needed = len(list_not_depr_males) - len(list_depr_fem)
extra_not_depr_fem_needed = len(list_not_depr_males) - len(list_not_depr_fem)

# Select original SBERT features for the four groups
df_orig_depr_males = X_sub_original[X_sub_original.index.isin(list_depr_males)]
df_orig_depr_fem = X_sub_original[X_sub_original.index.isin(list_depr_fem)]
df_orig_not_depr_fem = X_sub_original[X_sub_original.index.isin(list_not_depr_fem)]
df_orig_not_depr_males = X_sub_original[X_sub_original.index.isin(list_not_depr_males)]

# Select SBERT features from synthetic examples
df_generated_d_m = X_sub_generated[X_sub_generated['original_ID'].isin(list_depr_males)]
df_use_d_m = df_generated_d_m.head(extra_depr_males_needed)
df_d_m = pd.concat((df_orig_depr_males, df_use_d_m))

df_generated_d_f = X_sub_generated[X_sub_generated['original_ID'].isin(list_depr_fem)]
df_use_d_f = df_generated_d_f.head(extra_depr_fem_needed)
df_d_f = pd.concat((df_orig_depr_fem, df_use_d_f))

df_generated_nd_f = X_sub_generated[X_sub_generated['original_ID'].isin(list_not_depr_fem)]
df_use_nd_f = df_generated_nd_f.head(extra_not_depr_fem_needed)
df_nd_f = pd.concat((df_orig_not_depr_fem, df_use_nd_f))

# The symptom list of the depressive males
y_d_m = y_train_symptoms.loc[list_depr_males, :]
y_d_m_generated = y_train_symptoms.loc[df_use_d_m['original_ID'].values]
y_d_m_total = pd.concat((y_d_m, y_d_m_generated))

# The symptom list of the depressive females
y_d_f = y_train_symptoms.loc[list_depr_fem, :]
y_d_f_generated = y_train_symptoms.loc[df_use_d_f['original_ID'].values]
y_d_f_total = pd.concat((y_d_f, y_d_f_generated))

# The symptom list of the non-depressive females
y_nd_f = y_train_symptoms.loc[list_not_depr_fem, :]
y_nd_f_generated = y_train_symptoms.loc[df_use_nd_f['original_ID'].values]
y_nd_f_total = pd.concat((y_nd_f, y_nd_f_generated))

# The symptom list of the non-depressive males
y_nd_m_total = y_train_symptoms.loc[list_not_depr_males, :]

y_train_fold = pd.concat((y_d_m_total, y_d_f_total, y_nd_f_total, y_nd_m_total))

X_train_new = (pd.concat((df_d_m, df_d_f, df_nd_f, df_orig_not_depr_males)).
               drop('original_ID', level=0, axis=1))

_, _, pred_sev = sym_model(kelm_c=c, kelm_kernel=k, x_train=X_train_new, y_train=y_train_fold,
                           x_input=X_test)

df_symptom_results = pd.DataFrame({'pred_severity': pred_sev, 'true_severity': labels_test['PHQ_Score'].values,
                                   'true_binary': labels_test['PHQ_Binary'].values,
                                   'gender': labels_test['Gender'].values})

print_perf_measures(df_symptom_results)
