import os
import re
import csv
import pandas as pd
from pydub import AudioSegment


# Make sure that the folder path in the function below only contains transcript files for each participant

def get_split_audio_by_transcript(subset_dir: str) -> None:
    """
    :param subset_dir: path to the training, test or development folder
    :return: None
    """
    transcripts_path = os.path.join(subset_dir, 'transcripts/')
    audio_path = os.path.join(subset_dir, 'audio/')
    output_path = os.path.join(subset_dir, 'output/')

    files = os.listdir(transcripts_path)
    # files = files[:2]

    for file in files:
        subject = str(file)[:3]
        # find audio recording of participant
        audio_file_path = os.path.join(audio_path, subject + '_AUDIO.wav')  # path of audio file
        transcript_file_path = os.path.join(transcripts_path, file)  # path of recording file
        subject_output_folder_path = os.path.join(output_path, subject + '/')  # path to save split audio

        # Read audio file TODO dit hierheen verplaatst, misschien nog een keer checken
        audio = AudioSegment.from_wav(audio_file_path)

        # make folder for each participant separately
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        with open(transcript_file_path, 'r', encoding='utf-8') as csv_file:
            nr = 0  # for numbering the turns/splits of a participant

            reader = csv.reader(csv_file, delimiter='\t')
            next(reader)  # ignore title row
            for row in reader:
                if len(row) == 4:  # if not, the line may be empty
                    start_t = float(row[0])
                    end_t = float(row[1])
                    speaker = row[2]


                    if speaker == 'Participant':
                        segment = audio[start_t * 1000:end_t * 1000]  # get split audio from participant
                        formatted_nr = "{:04d}".format(nr)  # generate split audio str_functs

                        output_filename = (f"{os.path.splitext(os.path.basename(audio_file_path))[0]}"
                                           f"_{str(formatted_nr)}_{start_t}_{end_t}.wav")
                        nr += 1

                        # save audio segment
                        output_file_path = os.path.join(subject_output_folder_path, output_filename)
                        segment.export(output_file_path, format="wav")

            print(f'Done for {subject}')

    return


def make_index_file(subset_dir: str) -> None:
    """
    :param subset_dir: path to the training, test or development folder
    :return: None
    """
    # subset_path = training/test/development
    subset = os.path.basename(os.path.normpath(subset_dir))

    output_path = os.path.join(subset_dir, 'output/')
    transcripts_path = os.path.join(subset_dir, 'transcripts/')

    # locate and load csv file with PHQ8 answers per subject
    PHQ8_labels_filename = next((file for file in os.listdir(subset_dir) if "AVEC2017" in file), None)
    PHQ8_labels_file_path = os.path.join(subset_dir, PHQ8_labels_filename)
    df_with_PHQ8_labels = pd.read_csv(PHQ8_labels_file_path)

    # remove existing index file before adding information
    index_file_path = os.path.join(output_path, 'index_file_' + subset + '.csv')
    if os.path.exists(index_file_path):
        os.remove(index_file_path)

    # select only subject folders
    subject_folders = [f for f in os.listdir(output_path) if len(f) == 3]

    for subject in subject_folders:

        subject_folder_path = os.path.join(output_path, subject + '/')
        split_audio_files = os.listdir(subject_folder_path)  # a list of all the audio files per subject

        # locate and load transcript file for this subject
        transcript_file_path = os.path.join(transcripts_path, subject + '_TRANSCRIPT.csv')
        df_sub = pd.read_csv(transcript_file_path, delimiter='\t')
        participant_text = list(df_sub[df_sub['speaker'] == 'Participant']['value'])

        nr = 0
        # for each turn of this participant, write a row in the index file with information
        for audio_file in split_audio_files:
            # find duration of audio segment
            pattern = r"(\d+(\.\d+)?)_(\d+(\.\d+)?)"
            match = re.search(pattern, audio_file[15:-4])
            audio_duration = int(float(match.group(3)) * 1000 - float(match.group(1)) * 1000)  #

            audio_file_path = os.path.join(subject_folder_path, audio_file)

            with open(index_file_path, 'a', newline='', encoding='utf-8') as index_file:
                writer = csv.writer(index_file)
                writer.writerow([subject, audio_file, audio_file_path, participant_text[nr], audio_duration])
                nr += 1

    # add column names
    df_index_file = pd.read_csv(index_file_path, header=None)
    df_index_file.columns = ['Participant_ID', 'saved_file_name', 'file_path', 'text', 'duration']

    # merge index dataframe with PHQ8 labels
    merged_df = df_index_file.merge(df_with_PHQ8_labels, left_on='Participant_ID', right_on='Participant_ID')
    merged_df.to_csv(index_file_path, index=False)

    return


def preprocess_index_file(df_index_file: pd.DataFrame) -> pd.DataFrame:
    """
    :param df_index_file: dataframe of index file (each row is a turn from a participant
    :return: dataframe with turns of duration above 500 ms
    """
    df_turns_above_500 = df_index_file
    df_turns_above_500 = df_turns_above_500.query('duration > 500')
    nr_turns_per_subject = df_turns_above_500['Participant_ID'].value_counts()
    print(f'Average nr of turns over all participants: {nr_turns_per_subject.mean()}')

    duration_grouped = df_turns_above_500[['duration', 'Participant_ID']].groupby('Participant_ID')
    average_duration = duration_grouped.mean()  # average duration of turn per participant
    print(f'Average duration of turns over all participants: {average_duration.mean()}\n')

    return df_turns_above_500
