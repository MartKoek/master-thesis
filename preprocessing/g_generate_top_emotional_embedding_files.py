# TODO Wat heb ik hier voor nodig:
# TODO Start met training set
# TODO features van PDEM embedding
# TODO features van VAD embedding

import pandas as pd
import pickle
from constants import FUNCTIONALS, SBERT_MODELS
from evaluation.a_performance_metrics import print_perf_measures
from evaluation.utilities import plot_pred_trues, get_df_labels, save_results_per_ckf
from models import sym_model
from preprocessing.utils import generate_functional_df

def select_top_x_percent_arousal(group):
    num_rows = len(group)
    top_rows = group.sort_values(by='arousal_min_m', ascending=False).head(int(num_rows * percent))
    return top_rows

def select_top_x_percent_valence(group):
    num_rows = len(group)
    top_rows = group.sort_values(by='valence_min_m', ascending=False).head(int(num_rows * percent))
    return top_rows

def select_top_x_percent_dominance(group):
    num_rows = len(group)
    top_rows = group.sort_values(by='dominance_min_m', ascending=False).head(int(num_rows * percent))
    return top_rows


subject_vec_dev = pd.read_csv('C:/Users/mjkoe/Thesis/data/features/sbert_all-MiniLM-L12-v2/sbert_dev.csv', header=None)[0]
w2v_dev = pickle.load(open('C:/Users/mjkoe/Thesis/data/features/pdem_wav2vec/embedding_dev.pkl', 'rb'))
vad_dev = pickle.load(open('C:/Users/mjkoe/Thesis/data/features/pdem_vad/vad_dev.pkl', 'rb'))
vad_dev_abs = abs(vad_dev)

subject_vec_train = pd.read_csv('C:/Users/mjkoe/Thesis/data/features/sbert_all-MiniLM-L12-v2/sbert_train.csv', header=None)[0]
w2v_train = pickle.load(open('C:/Users/mjkoe/Thesis/data/features/pdem_wav2vec/embedding_train.pkl', 'rb'))
vad_train = pickle.load(open('C:/Users/mjkoe/Thesis/data/features/pdem_vad/vad_train.pkl', 'rb'))
vad_train_abs = abs(vad_train)

subject_vec_test = pd.read_csv('C:/Users/mjkoe/Thesis/data/features/sbert_all-MiniLM-L12-v2/sbert_test.csv', header=None)[0]
w2v_test = pickle.load(open('C:/Users/mjkoe/Thesis/data/features/pdem_wav2vec/embedding_test.pkl', 'rb'))
vad_test = pickle.load(open('C:/Users/mjkoe/Thesis/data/features/pdem_vad/vad_test.pkl', 'rb'))
vad_test_abs = abs(vad_test)

w2v_train.index = w2v_train.index.get_level_values(0)
vad_train_abs.index = vad_train_abs.index.get_level_values(0)
vad_train_abs['Participant_ID'] = subject_vec_train.values
vad_train_abs['arousal_min_m'] = abs(vad_train_abs['arousal'] -
                                     vad_train_abs.groupby('Participant_ID')['arousal'].transform('mean'))
vad_train_abs['dominance_min_m'] = abs(vad_train_abs['dominance'] -
                                     vad_train_abs.groupby('Participant_ID')['dominance'].transform('mean'))
vad_train_abs['valence_min_m'] = abs(vad_train_abs['valence'] -
                                     vad_train_abs.groupby('Participant_ID')['valence'].transform('mean'))

df_sorted_arousal_train = vad_train_abs.sort_values(by='arousal_min_m', ascending=False)
df_sorted_valence_train = vad_train_abs.sort_values(by='valence_min_m', ascending=False)
df_sorted_dominance_train = vad_train_abs.sort_values(by='dominance_min_m', ascending=False)

w2v_dev.index = w2v_dev.index.get_level_values(0)
vad_dev_abs.index = vad_dev_abs.index.get_level_values(0)
vad_dev_abs['Participant_ID'] = subject_vec_dev.values
vad_dev_abs['dominance_min_m'] = abs(vad_dev_abs['dominance'] -
                                     vad_dev_abs.groupby('Participant_ID')['dominance'].transform('mean'))

df_sorted_dominance_dev = vad_dev_abs.sort_values(by='dominance_min_m', ascending=False)


""" Test """
w2v_test.index = w2v_test.index.get_level_values(0)
vad_test_abs.index = vad_test_abs.index.get_level_values(0)
vad_test_abs['Participant_ID'] = subject_vec_test.values
vad_test_abs['arousal_min_m'] = abs(vad_test_abs['arousal'] -
                                     vad_test_abs.groupby('Participant_ID')['arousal'].transform('mean'))
vad_test_abs['dominance_min_m'] = abs(vad_test_abs['dominance'] -
                                     vad_test_abs.groupby('Participant_ID')['dominance'].transform('mean'))
vad_test_abs['valence_min_m'] = abs(vad_test_abs['valence'] -
                                     vad_test_abs.groupby('Participant_ID')['valence'].transform('mean'))

df_sorted_arousal_test = vad_test_abs.sort_values(by='arousal_min_m', ascending=False)
df_sorted_valence_test = vad_test_abs.sort_values(by='valence_min_m', ascending=False)
df_sorted_dominance_test = vad_test_abs.sort_values(by='dominance_min_m', ascending=False)

tops = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]

# Run this for every train, dev, and test, and for every VAD value (illustrated for training set of dominance

for percent in tops:
    # Apply the function to each participant group
    only_top_rows_train_dominance = (df_sorted_dominance_train.groupby('Participant_ID', group_keys=False)
                                   .apply(select_top_x_percent_dominance))
    rows_to_use_tr_dominance = only_top_rows_train_dominance.index
    part_vec_top_tr_dominance = only_top_rows_train_dominance['Participant_ID']

    w2v_train_top = w2v_train.loc[rows_to_use_tr_dominance]
    w2v_train_top.insert(0, 0, part_vec_top_tr_dominance.values)
    w2v_train_top.columns = range(len(w2v_train_top.columns))

    func = ["mean", "var", "std", "median", "quantile", "max"]
    df = w2v_train_top
    df.columns = df.columns - 1
    df = df.rename(columns={-1: 'Participant_ID'})
    df_functionals = df.groupby('Participant_ID').agg(func)

    # Overwrite the quantile column with 75 percentile
    if 'quantile' in func:
        df_quantile = df.groupby('Participant_ID').quantile(0.75)
        df_functionals.update({(i, f): df_quantile[i] for (i, f) in df_functionals.columns if f == 'quantile'})

    file_name = f'top_{percent}_emotional_train.csv'
    location_to_save = f'C:/Users/mjkoe/Thesis/data/features/wav2vec_top_emotional_func_dfs_dominance/{file_name}'
    df_functionals.to_csv(location_to_save)
