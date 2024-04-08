from sentence_transformers import SentenceTransformer
import pandas as pd
import os

from preprocessing.utils import generate_functional_df


def extract_sbert_embedding(path_data: str, paths_i_files: list, model: str, func: list) -> None:
    """
    :param func:
    :param path_data: path of data folder
    :param paths_i_files: paths of index files
    :param model: the name of the SBERT best_model
    :return: None

    Generates .csv files with SBERT embeddings of each turn of a participant
    """
    string_functionals = '_'.join(func)
    path_i_file_train, path_i_file_dev, path_i_file_test = paths_i_files
    transformer_model = SentenceTransformer(model)

    # Load index files
    train_data = pd.read_csv(path_i_file_train)[['Participant_ID', 'text']]
    dev_data = pd.read_csv(path_i_file_dev)[['Participant_ID', 'text']]
    test_data = pd.read_csv(path_i_file_test)[['Participant_ID', 'text']]

    # Check if the folder is already instantiated
    model_folder = os.path.join(path_data, f'features/sbert_{model}')  # TODO weer aanpassen naar gewoon features
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    # Generate the SBERT embeddings and column with participant ID's
    df_embed_train = pd.DataFrame(transformer_model.encode(train_data['text'].astype(str)))
    df_embed_train.insert(0, 'Participant_ID', train_data['Participant_ID'])

    df_embed_dev = pd.DataFrame(transformer_model.encode(dev_data['text'].astype(str)))
    df_embed_dev.insert(0, 'Participant_ID', dev_data['Participant_ID'])

    df_embed_test = pd.DataFrame(transformer_model.encode(test_data['text'].astype(str)))
    df_embed_test.insert(0, 'Participant_ID', test_data['Participant_ID'])

    # Make file saved_file_name and names
    train_SBERT_embed_file = os.path.join(model_folder, 'sbert_train.csv')
    dev_SBERT_embed_file = os.path.join(model_folder, 'sbert_dev.csv')
    test_SBERT_embed_file = os.path.join(model_folder, 'sbert_test.csv')

    df_embed_train.to_csv(train_SBERT_embed_file, header=False, index=False)
    print('Generated training set SBERT embeddings')
    df_embed_dev.to_csv(dev_SBERT_embed_file, header=False, index=False)
    print('Generated development set SBERT embeddings')
    df_embed_test.to_csv(test_SBERT_embed_file, header=False, index=False)
    print('Generated test set SBERT embeddings')

    # Make file saved_file_name and names
    train_functional_file = os.path.join(model_folder, f'sbert_train_{string_functionals}.csv')
    dev_functional_file = os.path.join(model_folder, f'sbert_dev_{string_functionals}.csv')
    test_functional_file = os.path.join(model_folder, f'sbert_test_{string_functionals}.csv')

    generate_functional_df(df=pd.read_csv(train_SBERT_embed_file, header=None),
                           func=func, saved_file_name=train_functional_file)
    generate_functional_df(df=pd.read_csv(dev_SBERT_embed_file, header=None),
                           func=func, saved_file_name=dev_functional_file)
    generate_functional_df(df=pd.read_csv(test_SBERT_embed_file, header=None),
                           func=func, saved_file_name=test_functional_file)

    return
