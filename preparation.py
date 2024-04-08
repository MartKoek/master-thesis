import os
import pandas as pd

from constants import DIR_TRAIN, DIR_TEST, DIR_DEV, SBERT_MODELS, FUNCTIONALS
from preprocessing.a_data_prep import get_split_audio_by_transcript, make_index_file, preprocess_index_file
from preprocessing.b_extract_VAD import extract_vad_pdem
from preprocessing.c_extract_PDEM_embeddings import extract_embedding_pdem
from preprocessing.d_extract_SBERT_embeddings import extract_sbert_embedding

""" 
Run this file only once:
-   Make audio split files
-   Make index file with split audio and labels
-   Preprocess the index file
-   Extract PDEM VAD scores
-   Extract PDEM embeddings
-   Extract SBERT embeddings for three different models
"""

directories = [DIR_TRAIN, DIR_DEV, DIR_TEST]

# create output folders inside data subset_path folders
for subset_path in directories:
    output_folder = os.path.join(subset_path, 'output/')

    # if the output folder did not exist, begin data prep
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        get_split_audio_by_transcript(subset_path)
        make_index_file(subset_path)

path_data = os.path.join(os.getcwd(), 'data/')

path_index_file_train = os.path.join(DIR_TRAIN, 'output/index_file_training.csv')
path_index_file_dev = os.path.join(DIR_DEV, 'output/index_file_development.csv')
path_index_file_test = os.path.join(DIR_TEST, 'output/index_file_test.csv')

path_clean_i_file_train = os.path.join(DIR_TRAIN, 'output/clean_i_file_training.csv')
path_clean_i_file_dev = os.path.join(DIR_DEV, 'output/clean_i_file_development.csv')
path_clean_i_file_test = os.path.join(DIR_TEST, 'output/clean_i_file_test.csv')

# preprocess index files
index_file_train = pd.read_csv(path_index_file_train)
index_file_dev = pd.read_csv(path_index_file_dev)
index_file_test = pd.read_csv(path_index_file_test)
print('Preprocess index files\nTraining set:')
clean_i_file_train = preprocess_index_file(index_file_train)
print('Development set:')
clean_i_file_dev = preprocess_index_file(index_file_dev)
print('Test set:')
clean_i_file_test = preprocess_index_file(index_file_test)

# Save the preprocessed index files
clean_i_file_train.to_csv(path_clean_i_file_train, index=False)
clean_i_file_dev.to_csv(path_clean_i_file_dev, index=False)
clean_i_file_test.to_csv(path_clean_i_file_test, index=False)

paths_index_processed = [path_clean_i_file_train, path_clean_i_file_dev, path_clean_i_file_test]

# Extract and save PDEM valence, arousal, dominance for each training, development, and test set
print('Extract PDEM scores: Valence, Arousal, Dominance')
extract_vad_pdem()

print('Extract PDEM scores: embeddings')
extract_embedding_pdem()

# Generate with each SBERT best_model the embeddings
for model in SBERT_MODELS:
    extract_sbert_embedding(paths_i_files=paths_index_processed, func=FUNCTIONALS, path_data=path_data, model=model)
