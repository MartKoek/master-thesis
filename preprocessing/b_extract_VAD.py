import audeer
import audonnx
import pandas as pd
import audinterface
import os
from preprocessing.utils import rename_file


# Below is a git-hub repository link of PDEM with the documentation for using PDEM
# https://github.com/audeering/w2v2-how-to?tab=readme-ov-file

def extract_vad_pdem():
    path_pdem_cache = 'C:/Users/mjkoe/Thesis/preprocessing/pdem_cache'
    path_model_cache = 'C:/Users/mjkoe/Thesis/preprocessing/model_cache'
    model_cache_root = audeer.mkdir(path_pdem_cache)
    model_root = audeer.mkdir(path_model_cache)
    root_dir = 'C:/Users/mjkoe/Thesis'

    cache_root = 'C:/Users/mjkoe/Thesis/data/features'  # Set the output files' root directory (pdem_vad scores)

    # Step 0: Download the PDEM best_model to the model_root dir and Load the PDEM best_model
    url = 'https://zenodo.org/record/6221127/files/w2v2-L-robust-12.6bc4a7fd-1.1.0.zip'
    archive_path = audeer.download_url(url, model_cache_root, verbose=True)
    audeer.extract_archive(archive_path, model_root)

    # Step 1: load best_model and set sampling rate
    model = audonnx.load(model_root)
    sampling_rate = 16000  # sampling rate of DAIC-WOZ is 16000

    # Define best_model interface, output is valence, arousal and dominance scores. This format is fixed
    logits = audinterface.Feature(
        model.labels('logits'), process_func=model,
        process_func_args={'outputs': 'logits'},
        sampling_rate=sampling_rate, resample=True,
        num_workers=8, verbose=True)  # optional to change num_workers: higher = faster

    # Step 2: Indicate the train, dev and test index files respectively which are prepared in 'a_data_prep'
    train_index_file = 'C:/Users/mjkoe/Thesis/data/training/output/clean_i_file_training.csv'
    dev_index_file = 'C:/Users/mjkoe/Thesis/data/development/output/clean_i_file_development.csv'
    test_index_file = 'C:/Users/mjkoe/Thesis/data/test/output/clean_i_file_test.csv'

    # Load index file
    daic_woz_train = pd.read_csv(train_index_file)
    daic_woz_dev = pd.read_csv(dev_index_file)
    daic_woz_test = pd.read_csv(test_index_file)

    # Setting the index to 'file_path' column: where audio recording is
    daic_woz_train.set_index('file_path', inplace=True)
    daic_woz_train.index = daic_woz_train.index.astype(str)
    daic_woz_dev.set_index('file_path', inplace=True)
    daic_woz_dev.index = daic_woz_dev.index.astype(str)
    daic_woz_test.set_index('file_path', inplace=True)
    daic_woz_test.index = daic_woz_test.index.astype(str)

    # Step 3: Extracting VAD scores using PDEM best_model for each wav
    """ Development set """
    logits.process_index(daic_woz_dev.index, root=root_dir,
                         # The output files will be saved here. File str_functs will be a random number ending with '.pkl'
                         cache_root=audeer.path(cache_root, 'pdem_vad/'))

    path_to_vad_folder = os.path.join(cache_root, 'pdem_vad/')
    files = os.listdir(path_to_vad_folder)
    for file in files:
        if file.endswith('.pkl') and not file[:3] == 'pdem_vad':  # check if the filename is a random number
            rename_file(os.path.join(path_to_vad_folder, file), 'vad_dev.pkl')

    """ Training set """
    logits.process_index(daic_woz_train.index, root=root_dir,
                         cache_root=audeer.path(cache_root, 'pdem_vad/'))

    files = os.listdir(path_to_vad_folder)
    for file in files:
        if file.endswith('.pkl') and not file[:3] == 'pdem_vad':
            rename_file(os.path.join(path_to_vad_folder, file), 'vad_train.pkl')

    """ Test set """
    logits.process_index(daic_woz_test.index, root=root_dir,
                         cache_root=audeer.path(cache_root, 'pdem_vad/'))

    files = os.listdir(path_to_vad_folder)
    for file in files:
        if file.endswith('.pkl') and not file[:3] == 'pdem_vad':
            rename_file(os.path.join(path_to_vad_folder, file), 'vad_test.pkl')

    return
