import os
import pandas as pd

def rename_file(old_path, new_name):
    try:
        directory, old_filename = os.path.split(old_path)
        new_path = os.path.join(directory, new_name)
        os.rename(old_path, new_path)
        print(f"File name successfully changed:{old_path} -> {new_path}")
    except OSError as e:
        print(f"Wrong with changing file name:{e}")


def generate_functional_df(df: pd.DataFrame, func: list, saved_file_name: str) -> None:
    """
    :param df: dataframe with embeddings
    :param func: list of functionals to calculate
    :param saved_file_name: location to store .csv file with summarized embeddings
    :return:
    """
    df.columns = df.columns - 1
    df = df.rename(columns={-1: 'Participant_ID'})
    df_functionals = df.groupby('Participant_ID').agg(func)

    # Overwrite the quantile column with 75 percentile
    if 'quantile' in func:
        df_quantile = df.groupby('Participant_ID').quantile(0.75)
        df_functionals.update({(i, f): df_quantile[i] for (i, f) in df_functionals.columns if f == 'quantile'})

    df_functionals.to_csv(saved_file_name)

    return
