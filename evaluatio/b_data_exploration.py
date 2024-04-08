import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from constants import DIR_TRAIN, DIR_TEST, DIR_DEV
import os

def generate_histograms_PHQ8(path_to_file: str) -> None:
    """
    :param path_to_file: path to *___*_split_Depression_AVEC2017.csv
    :return: None
    """
    df = pd.read_csv(path_to_file)
    name = os.path.basename(os.path.normpath(path_to_file))  # filename

    subset = name.split('_')[0]
    if subset == 'train':
        title_fig = 'Training set'
    if subset == 'dev':
        title_fig = 'Development set'

    # split the dataframe into different genders
    df_female = df.query('Gender == 0')
    df_male = df.query('Gender == 1')

    # set variables for the plots
    alpha_fig = .4
    gender_labels = ['Female', 'Male']
    gender_colors = ['orange', 'blue']

    # Generate plot for total PHQ8 score
    plt.hist([df_female['PHQ8_Score'], df_male['PHQ8_Score']], bins=range(0, 25), label=gender_labels,
             color=gender_colors, alpha=alpha_fig, edgecolor='black')

    plt.title(title_fig)
    plt.xlabel('PHQ8_Score')
    plt.ylabel('Frequency')
    plt.xticks(np.arange(.4, 25, 2), np.arange(0, 26, 2))  # the x-axis (score) can be 0 to 24
    plt.axvline(x=10, color='r', lw=1, label='Depression threshold')  # set threshold for a depression label
    plt.ylim(0, 9.5)  # set frequency
    plt.legend()
    plt.show()

    # Generate plot for the binary labels
    plt.hist([df_female['PHQ8_Binary'], df_male['PHQ8_Binary']], bins=range(3), label=gender_labels,
             color=gender_colors, alpha=alpha_fig, edgecolor='black')

    plt.title(title_fig)
    plt.xlabel('Depressed')
    plt.ylabel('Frequency')
    plt.xticks([.5, 1.5], ['No', 'Yes'])
    plt.legend()
    plt.show()

    # generate plots for each symptom separately
    fig, axs = plt.subplots(3, 3, figsize=(11, 6))
    fig.subplots_adjust(hspace=0.5)

    for i, symptom in enumerate(['PHQ8_NoInterest', 'PHQ8_Depressed', 'PHQ8_Sleep', 'PHQ8_Tired',
                                 'PHQ8_Appetite', 'PHQ8_Failure', 'PHQ8_Concentrating', 'PHQ8_Moving']):
        row, col = divmod(i, 3)
        ax = axs[row, col]

        ax.hist([df_female[symptom], df_male[symptom]], bins=range(5), label=gender_labels, color=gender_colors,
                alpha=alpha_fig, edgecolor='black')
        ax.set_xlabel(symptom[5:])

        # set the y limits appropriately
        if subset == 'train':
            ax.set_ylim(0, 50)
        if subset == 'dev':
            ax.set_ylim(0, 15)

        ax.set_xticks([.5, 1.5, 2.5, 3.5], [0, 1, 2, 3])
        ax.set_ylabel('Frequency')

    # Create a legend for the entire figure
    fig.legend(gender_labels, loc='lower right', title='Gender', bbox_to_anchor=(0.8, 0.20))

    fig.delaxes(axs[2][2])  # Remove the 9th figure
    fig.subplots_adjust(wspace=0.25, hspace=.35, top=.92)  # set spacing of sub figures
    fig.suptitle(title_fig)  # add large title above figure
    plt.show()

    return


def generate_df_ellie_per_subject(subset_path: str):
    """
    :param subset_path: path to the folder of data subset
    :return:
    """
    transcripts_path = os.path.join(subset_path, 'transcripts/')
    files = os.listdir(transcripts_path)
    subjects = [sub[:3] for sub in files]

    ellies_turns = []
    for file in files:
        transcript_file_path = os.path.join(transcripts_path, file)  # path of transcript file
        df_subject = pd.read_csv(transcript_file_path, delimiter='\t')
        text_ellie = list(df_subject[df_subject['speaker'] == 'Ellie']['value'])  # a list with all ellies turns
        ellies_turns.append(text_ellie)

    # make dataframe with ellies turns in each interview
    ellies_turns_df = pd.DataFrame(ellies_turns, index=subjects)
    stacked_df = ellies_turns_df.stack()
    unique_turns_ellie = stacked_df.unique()

    list_vectors = []
    for sub in subjects:
        sub_vec_turns = [1 if i in list(ellies_turns_df.loc[sub]) else 0 for i in unique_turns_ellie]
        list_vectors.append(sub_vec_turns)

    df_turns_per_subject = pd.concat([pd.DataFrame(unique_turns_ellie), pd.DataFrame(list_vectors).T], axis=1)
    new_header = subjects
    new_header.insert(0, 'text')
    df_turns_per_subject.columns = new_header

    return df_turns_per_subject


def explore_df(df_turns):
    # for these participants there are no ellie turns recorded
    for subject in ['451', '458', '480']:
        if subject in df_turns.columns:
            df_turns = df_turns.drop([subject], axis=1)

    # Use findall to extract all occurrences of the pattern (...) in the input string
    clean_turns = []
    for turn in df_turns['text']:
        text_between_br = re.findall(r'\((.*?)\)', turn)
        if not text_between_br:
            cleaned = turn.strip()
        else:
            cleaned = text_between_br[0].strip()
        cleaned = cleaned.replace('(', '').replace(')', '').replace('<', '').replace('>', '').replace('[', '').replace(
            ']', '')
        clean_turns.append(cleaned)

    df_turns['text'] = clean_turns
    # merge the same turns together and sum the occurrences over participants
    df_merged_sim_turns = df_turns.groupby(['text'], as_index=False).agg('sum')  # this also orders alphabetically

    for index, row in df_merged_sim_turns.iterrows():
        if len(row['text'].split()) == 1 or len(row['text'].split()) == 2:  # remove short responses
            df_merged_sim_turns.drop(index, inplace=True)
        if "hi i'm" in row['text']:  # remove introduction
            df_merged_sim_turns.drop(index, inplace=True)
        if "oh my" in row['text']:  # remove response oh my gosh
            df_merged_sim_turns.drop(index, inplace=True)
        if "uh huh uh huh" in row['text']:  # remove response uh huh uh huh
            df_merged_sim_turns.drop(index, inplace=True)

    # get number of occurrences over all participants
    df_turns_summed = pd.DataFrame({'sum': df_merged_sim_turns.sum(axis=1, numeric_only=True)})
    df_turns_summed.insert(0, 'text', df_merged_sim_turns['text'])

    df_turns_summed_ord = df_turns_summed.sort_values(by='sum', ascending=False)
    print(df_turns_summed_ord.head(20))

    return df_merged_sim_turns


# specify these directories in constants.py
directories = [DIR_TRAIN, DIR_DEV, DIR_TEST]

""" Explore depression label and symptoms per gender """
lbls_file_tr = os.path.join(directories[0], 'train_split_Depression_AVEC2017.csv')
lbls_file_dev = os.path.join(directories[1], 'dev_split_Depression_AVEC2017.csv')
generate_histograms_PHQ8(lbls_file_tr)
generate_histograms_PHQ8(lbls_file_dev)

""" Explore Ellie's turns """
df_train = generate_df_ellie_per_subject(directories[0])
df_dev = generate_df_ellie_per_subject(directories[1])
