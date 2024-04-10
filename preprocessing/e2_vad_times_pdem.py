import pandas as pd
import pickle
from constants import FUNCTIONALS, SBERT_MODELS
from evaluation.a_performance_metrics import performance_measures
from evaluation.utilities import plot_pred_trues, get_df_labels, save_results_per_ckf
from models import sym_model
from preprocessing.utils import generate_functional_df

string_functionals = '_'.join(FUNCTIONALS)

# Find labels of training, development, test set
labels_train, labels_dev, labels_test = get_df_labels()

train_symptoms = labels_train.drop(columns=['PHQ8_Binary', 'PHQ8_Score', 'Gender']).to_numpy()
dev_symptoms = labels_dev.drop(columns=['PHQ8_Binary', 'PHQ8_Score', 'Gender']).to_numpy()

dev_binary_vec = labels_dev['PHQ8_Binary']
test_binary_vec = labels_test['PHQ_Binary']

dev_severity_vec = labels_dev['PHQ8_Score']
test_severity_vec = labels_test['PHQ_Score']

i_dev = pd.read_csv('C:/Users/mjkoe/Thesis/data/development/output/clean_i_file_development.csv')

participant_vec_dev = pd.read_csv('C:/Users/mjkoe/Thesis/data/features/sbert_all-MiniLM-L12-v2/sbert_dev.csv', header=None)[0]
participant_vec_train = pd.read_csv('C:/Users/mjkoe/Thesis/data/features/sbert_all-MiniLM-L12-v2/sbert_train.csv', header=None)[0]
participant_vec_test = pd.read_csv('C:/Users/mjkoe/Thesis/data/features/sbert_all-MiniLM-L12-v2/sbert_test.csv', header=None)[0]

with open('C:/Users/mjkoe/Thesis/data/features/pdem_wav2vec/embedding_dev.pkl', 'rb') as f:
    w2v_dev = pickle.load(f)
with open('C:/Users/mjkoe/Thesis/data/features/pdem_wav2vec/embedding_train.pkl', 'rb') as f:
    w2v_train = pickle.load(f)
with open('C:/Users/mjkoe/Thesis/data/features/pdem_wav2vec/embedding_test.pkl', 'rb') as f:
    w2v_test = pickle.load(f)

with open('C:/Users/mjkoe/Thesis/data/features/pdem_vad/vad_dev.pkl', 'rb') as f:
    vad_dev = pickle.load(f)
with open('C:/Users/mjkoe/Thesis/data/features/pdem_vad/vad_train.pkl', 'rb') as f:
    vad_train = pickle.load(f)
with open('C:/Users/mjkoe/Thesis/data/features/pdem_vad/vad_test.pkl', 'rb') as f:
    vad_test = pickle.load(f)


for single_vad in vad_dev.columns:

    weighted_dev = w2v_dev.mul(vad_dev[single_vad].values, axis=0)
    weighted_dev.insert(0, 0, participant_vec_dev.values)
    weighted_dev.columns = range(len(weighted_dev.columns))
    loc_dev_w = f'C:/Users/mjkoe/Thesis/data/features/pdem_weighted/with_{single_vad}_dev_functs.csv'
    generate_functional_df(weighted_dev,func=FUNCTIONALS,saved_file_name=loc_dev_w)

    weighted_dev = w2v_dev.mul(abs(vad_dev[single_vad].values), axis=0)
    weighted_dev.insert(0, 0, participant_vec_dev.values)
    weighted_dev.columns = range(len(weighted_dev.columns))
    loc_dev_w = f'C:/Users/mjkoe/Thesis/data/features/pdem_weighted/with_{single_vad}_abs_dev_functs.csv'
    generate_functional_df(weighted_dev,func=FUNCTIONALS,saved_file_name=loc_dev_w)

    weighted_train = w2v_train.mul(vad_train[single_vad].values, axis=0)
    weighted_train.insert(0, 0, participant_vec_train.values)
    weighted_train.columns = range(len(weighted_train.columns))
    loc_train_w = f'C:/Users/mjkoe/Thesis/data/features/pdem_weighted/with_{single_vad}_train_functs.csv'
    generate_functional_df(weighted_train,func=FUNCTIONALS,saved_file_name=loc_train_w)

    weighted_train = w2v_train.mul(abs(vad_train[single_vad].values), axis=0)
    weighted_train.insert(0, 0, participant_vec_train.values)
    weighted_train.columns = range(len(weighted_train.columns))
    loc_train_w = f'C:/Users/mjkoe/Thesis/data/features/pdem_weighted/with_{single_vad}_abs_train_functs.csv'
    generate_functional_df(weighted_train,func=FUNCTIONALS,saved_file_name=loc_train_w)

    weighted_test = w2v_test.mul(vad_test[single_vad].values, axis=0)
    weighted_test.insert(0, 0, participant_vec_test.values)
    weighted_test.columns = range(len(weighted_test.columns))
    loc_test_w = f'C:/Users/mjkoe/Thesis/data/features/pdem_weighted/with_{single_vad}_test_functs.csv'
    generate_functional_df(weighted_test,func=FUNCTIONALS,saved_file_name=loc_test_w)

    weighted_test = w2v_test.mul(abs(vad_test[single_vad].values), axis=0)
    weighted_test.insert(0, 0, participant_vec_test.values)
    weighted_test.columns = range(len(weighted_test.columns))
    loc_test_w = f'C:/Users/mjkoe/Thesis/data/features/pdem_weighted/with_{single_vad}_abs_test_functs.csv'
    generate_functional_df(weighted_test,func=FUNCTIONALS,saved_file_name=loc_test_w)
