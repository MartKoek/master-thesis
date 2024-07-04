import pandas as pd
from evaluation.utilities import get_df_labels
import itertools
import pickle
import numpy as np
from constants import FUNCTIONALS

# A list of all possible combinations of functionals
str_functs = '_'.join(FUNCTIONALS)
list_combos_functionals = []
for index in range(1, len(FUNCTIONALS)):
    for list_of_functionals in itertools.combinations(FUNCTIONALS, index):
        list_combos_functionals.append(list(list_of_functionals))


def generate_functional_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    :param df: dataframe with embeddings
    :return: the row with functionals
    """
    func = ["mean", "var", "std", "median", "quantile", "max"]
    df.columns = np.array(range(len(df.columns))) - 1
    df = df.rename(columns={-1: 'ID'})
    df_functionals = df.groupby('ID').agg(func)

    # Overwrite the quantile column with 75 percentile
    if 'quantile' in func:
        df_quantile = df.groupby('ID').quantile(0.75)
        df_functionals.update({(i, f): df_quantile[i] for (i, f) in df_functionals.columns if f == 'quantile'})

    return df_functionals


# Find labels of training, development, test set
labels_train, labels_dev, _ = get_df_labels()
labels_total = pd.concat((labels_train, labels_dev))
gender_vec_total = labels_total['Gender']
binary_vec_total = labels_total['PHQ8_Binary']
severity_vec_total = labels_total['PHQ8_Score']
symptoms_vec_total = labels_total.drop(columns=['PHQ8_Binary', 'PHQ8_Score', 'Gender'])

subjects_dep1_male1 = list(labels_total.query('PHQ8_Binary == 1 & Gender == 1').index)
subjects_dep1_fem0 = list(labels_total.query('PHQ8_Binary == 1 & Gender == 0').index)
subjects_nondep0_male1 = list(labels_total.query('PHQ8_Binary == 0 & Gender == 1').index)
subjects_nondep0_fem0 = list(labels_total.query('PHQ8_Binary == 0 & Gender == 0').index)

index_dev = pd.read_csv('location of index file development')['Participant_ID']
index_train = pd.read_csv('location of training file')['Participant_ID']
index_train_dev = pd.concat((index_train, index_dev))

""" For SBERT """


def generate_SBERT_samples():
    loc_train = ''  # location of sentence-embedding file of training set
    loc_dev = ''  # location of sentence-embedding file of development set
    SBERT_rows_train = pd.read_csv(loc_train)
    SBERT_rows_dev = pd.read_csv(loc_dev)
    SBERT_train_dev = pd.concat((SBERT_rows_train, SBERT_rows_dev))

    df_dep1_fem0 = SBERT_train_dev[SBERT_train_dev[0].isin(subjects_dep1_fem0)]
    df_nondep0_fem0 = SBERT_train_dev[SBERT_train_dev[0].isin(subjects_nondep0_fem0)]
    df_dep1_male1 = SBERT_train_dev[SBERT_train_dev[0].isin(subjects_dep1_male1)]
    df_nondep0_male1 = SBERT_train_dev[SBERT_train_dev[0].isin(subjects_nondep0_male1)]

    for subject in subjects_dep1_male1:
        df_subject = SBERT_train_dev.loc[SBERT_train_dev[0] == subject]
        rows_per_part = len(df_subject) // 3

        # Slice the DataFrame into three parts
        part1 = df_subject.iloc[:rows_per_part].copy()
        part2 = df_subject.iloc[rows_per_part:2*rows_per_part].copy()
        part3 = df_subject.iloc[2*rows_per_part:].copy()

        part1[0] = part1[0].apply(lambda x: str(x) + 'A')
        part2[0] = part2[0].apply(lambda x: str(x) + 'B')
        part3[0] = part3[0].apply(lambda x: str(x) + 'C')

        df_dep1_male1 = df_dep1_male1._append(pd.concat((part1, part2, part3)), ignore_index=True)

    for subject in subjects_dep1_fem0:
        df_subject = SBERT_train_dev.loc[SBERT_train_dev[0] == subject]
        rows_per_part = len(df_subject) // 2

        # Slice the DataFrame into three parts
        part1 = df_subject.iloc[:rows_per_part].copy()
        part2 = df_subject.iloc[rows_per_part:].copy()

        part1[0] = part1[0].apply(lambda x: str(x) + 'A')
        part2[0] = part2[0].apply(lambda x: str(x) + 'B')

        df_dep1_fem0 = df_dep1_fem0._append(pd.concat((part1, part2)), ignore_index=True)

    for subject in subjects_nondep0_fem0:
        df_subject = SBERT_train_dev.loc[SBERT_train_dev[0] == subject]
        rows_per_part = len(df_subject) // 4

        # Slice the DataFrame into three parts
        part1 = df_subject.iloc[rows_per_part:3*rows_per_part].copy()

        part1[0] = part1[0].apply(lambda x: str(x) + 'A')

        df_nondep0_fem0 = df_nondep0_fem0._append(part1, ignore_index=True)

    new_total = pd.concat((df_dep1_male1, df_dep1_fem0, df_nondep0_fem0, df_nondep0_male1))
    new_total_functs = generate_functional_df(new_total)

    new_total.to_csv('location of file to save (original + generated) samples (sentence-level)')
    new_total_functs.to_csv('location to save session-level embeddings ')


""" For PDEM wav2vec """


def generate_PDEM_samples():
    PDEM_rows_dev = pickle.load(open('C:/Users/mjkoe/Thesis/data/features/pdem_wav2vec/embedding_dev.pkl', 'rb'))
    PDEM_rows_train = pickle.load(open('C:/Users/mjkoe/Thesis/data/features/pdem_wav2vec/embedding_train.pkl', 'rb'))

    PDEM_train_dev = pd.concat((PDEM_rows_train, PDEM_rows_dev))
    PDEM_train_dev.insert(0, 0, index_train_dev.values)
    PDEM_train_dev = PDEM_train_dev.reset_index(drop=True)

    df_dep1_fem0 = PDEM_train_dev[PDEM_train_dev[0].isin(subjects_dep1_fem0)]
    df_nondep0_fem0 = PDEM_train_dev[PDEM_train_dev[0].isin(subjects_nondep0_fem0)]
    df_dep1_male1 = PDEM_train_dev[PDEM_train_dev[0].isin(subjects_dep1_male1)]
    df_nondep0_male1 = PDEM_train_dev[PDEM_train_dev[0].isin(subjects_nondep0_male1)]

    for subject in subjects_dep1_male1:
        df_subject = PDEM_train_dev.loc[PDEM_train_dev[0] == subject]
        rows_per_part = len(df_subject) // 3

        # Slice the DataFrame into three parts
        part1 = df_subject.iloc[:rows_per_part].copy()
        part2 = df_subject.iloc[rows_per_part:2*rows_per_part].copy()
        part3 = df_subject.iloc[2*rows_per_part:].copy()

        part1[0] = part1[0].apply(lambda x: str(x) + 'A')
        part2[0] = part2[0].apply(lambda x: str(x) + 'B')
        part3[0] = part3[0].apply(lambda x: str(x) + 'C')

        df_dep1_male1 = df_dep1_male1._append(pd.concat((part1, part2, part3)), ignore_index=True)

    for subject in subjects_dep1_fem0:
        df_subject = PDEM_train_dev.loc[PDEM_train_dev[0] == subject]
        rows_per_part = len(df_subject) // 2

        # Slice the DataFrame into three parts
        part1 = df_subject.iloc[:rows_per_part].copy()
        part2 = df_subject.iloc[rows_per_part:].copy()

        part1[0] = part1[0].apply(lambda x: str(x) + 'A')
        part2[0] = part2[0].apply(lambda x: str(x) + 'B')

        df_dep1_fem0 = df_dep1_fem0._append(pd.concat((part1, part2)), ignore_index=True)

    for subject in subjects_nondep0_fem0:
        df_subject = PDEM_train_dev.loc[PDEM_train_dev[0] == subject]
        rows_per_part = len(df_subject) // 4

        # Slice the DataFrame into three parts
        part1 = df_subject.iloc[rows_per_part:3*rows_per_part].copy()

        part1[0] = part1[0].apply(lambda x: str(x) + 'A')

        df_nondep0_fem0 = df_nondep0_fem0._append(part1, ignore_index=True)

    new_total = pd.concat((df_dep1_male1, df_dep1_fem0, df_nondep0_fem0, df_nondep0_male1))
    new_total_functs = generate_functional_df(new_total)

    new_total.to_csv('location of file to save (original + generated) samples (sentence-level)')
    new_total_functs.to_csv('location to save session-level embeddings ')


generate_SBERT_samples()
generate_PDEM_samples()
