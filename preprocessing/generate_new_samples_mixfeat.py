import pandas as pd
import numpy as np
from evaluation.utilities import get_df_labels
import itertools
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
    df.columns = list(range(len(df.columns)))
    df.columns = df.columns - 1
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

index_dev = pd.read_csv('location index file dev')['Participant_ID']
index_train = pd.read_csv('location index file train')['Participant_ID']
index_train_dev = pd.concat((index_train, index_dev))


""" For SBERT or PDEM (change file name in beginning) """


def generate_samples():

    SBERT_train = pd.read_csv('C:/Users/mjkoe/Thesis/data/features/pdem_wav2vec/'
                              'w2v_train_mean_var_std_median_quantile_max.csv',
                              header=[0, 1], skipinitialspace=True, index_col=0)
    SBERT_dev = pd.read_csv('C:/Users/mjkoe/Thesis/data/features/pdem_wav2vec/'
                            'w2v_dev_mean_var_std_median_quantile_max.csv',
                            header=[0, 1], skipinitialspace=True, index_col=0)
    SBERT_train_dev = pd.concat((SBERT_train, SBERT_dev))

    df_dep1_fem0 = SBERT_train_dev[SBERT_train_dev.index.isin(subjects_dep1_fem0)]
    df_nondep0_fem0 = SBERT_train_dev[SBERT_train_dev.index.isin(subjects_nondep0_fem0)]
    df_dep1_male1 = SBERT_train_dev[SBERT_train_dev.index.isin(subjects_dep1_male1)]
    df_nondep0_male1 = SBERT_train_dev[SBERT_train_dev.index.isin(subjects_nondep0_male1)]

    print(len(subjects_nondep0_male1))

    for i in range(len(subjects_dep1_male1)-3):
        df_subject1 = SBERT_train_dev.loc[subjects_dep1_male1[i]]
        df_subject2 = SBERT_train_dev.loc[subjects_dep1_male1[i + 1]]
        df_subject3 = SBERT_train_dev.loc[subjects_dep1_male1[i + 2]]
        df_subject4 = SBERT_train_dev.loc[subjects_dep1_male1[i + 3]]

        lamb_list = np.random.normal(0.5, 0.1, 3)
        lamb_list = np.where((lamb_list > 1) | (lamb_list < 0), 0.5, lamb_list)

        DM01 = lamb_list[0] * df_subject1 + (1-lamb_list[0]) * df_subject2
        DM02 = lamb_list[1] * df_subject1 + (1-lamb_list[1]) * df_subject3
        DM03 = lamb_list[2] * df_subject1 + (1-lamb_list[2]) * df_subject4

        new = pd.concat([DM01, DM02, DM03], axis=1).transpose()
        new.index = [str(subjects_dep1_male1[i])+'A', str(subjects_dep1_male1[i])+'B', str(subjects_dep1_male1[i])+'C']
        df_dep1_male1 = pd.concat([df_dep1_male1, new], ignore_index=False)

    for i in range(len(subjects_dep1_fem0)-2):
        df_subject1 = SBERT_train_dev.loc[subjects_dep1_fem0[i]]
        df_subject2 = SBERT_train_dev.loc[subjects_dep1_fem0[i + 1]]
        df_subject3 = SBERT_train_dev.loc[subjects_dep1_fem0[i + 2]]

        lamb_list = np.random.normal(0.5, 0.1, 2)
        lamb_list = np.where((lamb_list > 1) | (lamb_list < 0), 0.5, lamb_list)

        DM01 = lamb_list[0] * df_subject1 + (1-lamb_list[0]) * df_subject2
        DM02 = lamb_list[1] * df_subject1 + (1-lamb_list[1]) * df_subject3

        new = pd.concat([DM01, DM02], axis=1).transpose()
        new.index = [str(subjects_dep1_fem0[i])+'A', str(subjects_dep1_fem0[i])+'B']
        df_dep1_fem0 = pd.concat([df_dep1_fem0, new], ignore_index=False)

    for i in range(len(subjects_nondep0_fem0)-1):
        df_subject1 = SBERT_train_dev.loc[subjects_nondep0_fem0[i]]
        df_subject2 = SBERT_train_dev.loc[subjects_nondep0_fem0[i + 1]]

        lamb_list = np.random.normal(0.5, 0.1, 1)
        lamb_list = np.where((lamb_list > 1) | (lamb_list < 0), 0.5, lamb_list)

        DM01 = lamb_list[0] * df_subject1 + (1-lamb_list[0]) * df_subject2

        new = pd.DataFrame(DM01).transpose()

        new.index = [str(subjects_nondep0_fem0[i])+'A']
        df_nondep0_fem0 = pd.concat([df_nondep0_fem0, new], ignore_index=False)

    new_total = pd.concat((df_dep1_male1, df_dep1_fem0, df_nondep0_fem0, df_nondep0_male1))

    new_total.to_csv('C:/Users/mjkoe/Thesis/data/features/pdem_wav2vec/w2v_train_dev_mixfeat_functs.csv')
