import audeer
import audonnx
import pandas as pd
import audinterface
import os
from preprocessing.utils import rename_file


# Below is a git-hub repository link of PDEM with the documentation for using PDEM
# https://github.com/audeering/w2v2-how-to?tab=readme-ov-file

def extract_embedding_pdem():
    path_pdem_cache = 'C:/Users/mjkoe/Thesis/preprocessing/pdem_cache'
    path_model_cache = 'C:/Users/mjkoe/Thesis/preprocessing/model_cache'
    audeer.mkdir(path_pdem_cache)
    model_root = audeer.mkdir(path_model_cache)
    root_dir = 'C:/Users/mjkoe/Thesis'

    cache_root = 'C:/Users/mjkoe/Thesis/data/features'  # Set the output files' root directory (feature embed scores)

    # Step 0: Download the PDEM best_model to the model_root dir and Load the PDEM best_model, this has been done in b_extract_vad

    # Step 1: load best_model and set sampling rate
    model = audonnx.load(model_root)
    sampling_rate = 16000  # sampling rate of DAIC-WOZ is 16000

    # Define best_model interface, output is PDEM embedding. This format is fixed defined by audinterface package.
    hidden_states = audinterface.Feature(
        model.labels('hidden_states'), process_func=model,
        process_func_args={'outputs': 'hidden_states'},
        sampling_rate=sampling_rate, resample=True,
        num_workers=8, verbose=True)

    # Load index file
    daic_woz_train = pd.read_csv('C:/Users/mjkoe/Thesis/data/training/output/clean_i_file_training.csv')
    daic_woz_dev = pd.read_csv('C:/Users/mjkoe/Thesis/data/development/output/clean_i_file_development.csv')
    daic_woz_test = pd.read_csv('C:/Users/mjkoe/Thesis/data/test/output/clean_i_file_test.csv')

    # Setting the index to 'file_path' column: where audio recording is
    daic_woz_train.set_index('file_path', inplace=True)
    daic_woz_train.index = daic_woz_train.index.astype(str)
    daic_woz_dev.set_index('file_path', inplace=True)
    daic_woz_dev.index = daic_woz_dev.index.astype(str)
    daic_woz_test.set_index('file_path', inplace=True)
    daic_woz_test.index = daic_woz_test.index.astype(str)

    # Step 3: Extracting VAD scores using PDEM best_model for each wav
    """ Development set """
    hidden_states.process_index(daic_woz_dev.index, root=root_dir,
                                cache_root=audeer.path(cache_root, 'pdem_wav2vec/'))

    path_to_pdem_folder = os.path.join(cache_root, 'pdem_wav2vec/')
    files = os.listdir(path_to_pdem_folder)
    for file in files:
        if file.endswith('.pkl') and not file[:5] == 'pdem_wav2vec':  # check if the filename is a random number
            rename_file(os.path.join(path_to_pdem_folder, file), 'embedding_dev.pkl')

    """ Training set """
    hidden_states.process_index(daic_woz_train.index, root=root_dir,
                                cache_root=audeer.path(cache_root, 'pdem_wav2vec/'))

    files = os.listdir(path_to_pdem_folder)
    for file in files:
        if file.endswith('.pkl') and not file[:5] == 'pdem_wav2vec':
            rename_file(os.path.join(path_to_pdem_folder, file), 'embedding_train.pkl')

    """ Test set """
    hidden_states.process_index(daic_woz_test.index, root=root_dir,
                                cache_root=audeer.path(cache_root, 'pdem_wav2vec/'))

    files = os.listdir(path_to_pdem_folder)
    for file in files:
        if file.endswith('.pkl') and not file[:5] == 'pdem_wav2vec':
            rename_file(os.path.join(path_to_pdem_folder, file), 'embedding_test.pkl')

    return
