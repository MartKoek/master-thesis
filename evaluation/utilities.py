import pandas as pd
from constants import FUNCTIONALS
import itertools
import matplotlib.pyplot as plt

from evaluation.a_performance_metrics import non_binary_metrics
from models import sym_model

string_functionals = '_'.join(FUNCTIONALS)
list_combos_functionals = []
for index in range(1, len(FUNCTIONALS)):
    for list_of_functionals in itertools.combinations(FUNCTIONALS, index):
        list_combos_functionals.append(list(list_of_functionals))


# def get_preds_feature_fusion(kelm_c: float, kelm_kernel: str, x_train: pd.DataFrame,
#                              y_train: np.array, x_input: pd.DataFrame):
#     # Scale and normalize the SBERT embeddings
#     scaler = StandardScaler().fit(x_train)
#
#     X_train_scaled = normalize(scaler.transform(x_train))
#     X_input_scaled = normalize(scaler.transform(x_input))
#
#     elm = ELM(c=kelm_c, kernel=kelm_kernel, is_classification=False, weighted=False)
#
#     elm.fit(X_train_scaled, y_train)
#     pred_symptoms = elm.predict(X_input_scaled)
#
#     # Mapping predictions to original scale
#     pred_symptoms = (pred_symptoms * np.std(y_train, axis=0)) + np.mean(y_train, axis=0)
#
#     # Sanitizing symptoms
#     pred_symptoms[pred_symptoms < 0] = 0
#     pred_symptoms[pred_symptoms > 3] = 3
#
#     # Rounding symptoms to nearest integer
#     raw_symptoms = pred_symptoms
#     pred_symptoms = np.rint(pred_symptoms)
#     pred_severity = np.sum(pred_symptoms, axis=1)  # sum the symptom prediction scores to total severity
#
#     return raw_symptoms, pred_symptoms, pred_severity


def plot_pred_trues(df_pred):
    plt.scatter(df_pred.query('gender == 0')['true_severity'], df_pred.query('gender == 0')['pred_severity'],
                color='r', label='0 = female')
    plt.scatter(df_pred.query('gender == 1')['true_severity'], df_pred.query('gender == 1')['pred_severity'],
                color='b', label='1 = male')
    plt.ylim(-1, 25)
    plt.xlim(-1, 25)
    plt.xlabel('True severity')
    plt.ylabel('Predicted severity ')
    plt.axvline(x=9.50, c='black', lw=.5)
    plt.legend()
    plt.axhline(y=9.50, c='black', lw=.5)
    plt.show()
    return


def get_df_labels() -> tuple:
    labels_train = pd.read_csv('C:/Users/mjkoe/Thesis/data/training/train_split_Depression_AVEC2017.csv',
                               index_col='Participant_ID')
    labels_dev = pd.read_csv('C:/Users/mjkoe/Thesis/data/development/dev_split_Depression_AVEC2017.csv',
                             index_col='Participant_ID')
    labels_test = pd.read_csv('C:/Users/mjkoe/Thesis/data/test/test_split_Depression_AVEC2017.csv',
                              index_col='Participant_ID')
    return labels_train, labels_dev, labels_test


def save_results_per_ckf(models, train_symptoms, input_severity_vec):
    c_list = [10, 15, 20, 25, 30, 40, 50, 60, 65, 70, 75, 80, 90, 100]
    kernels_list = ['sigmoid', 'linear', 'poly', 'rbf']

    results = []
    for model in models:
        print(f'Model: {model}')

        if 'sbert' in model:
            pre = 'sbert'
        if 'vad' in model:
            pre = 'vad'
        if 'wav2vec' in model:
            pre = 'embed'

        X_train_model = pd.read_csv(
            f'C:/Users/mjkoe/Thesis/data/features/{model}/{pre}_train_{string_functionals}.csv',
            header=[0, 1], skipinitialspace=True, index_col=0)
        X_dev_model = pd.read_csv(
            f'C:/Users/mjkoe/Thesis/data/features/{model}/{pre}_dev_{string_functionals}.csv',
            header=[0, 1], skipinitialspace=True, index_col=0)

        for list_of_funcs in list_combos_functionals:
            for c in c_list:
                for k in kernels_list:
                    raw_sym_ckf, pred_sym_ckf, pred_sev_ckf = sym_model(kelm_c=c, kelm_kernel=k,
                                                                        x_train=X_train_model,
                                                                        y_train=train_symptoms,
                                                                        x_input=X_dev_model)

                    rmse, ccc, mae = non_binary_metrics(trues=input_severity_vec, preds=pred_sev_ckf)

                    results.append(['kelm_symptoms', model, '_'.join(list_of_funcs), c, k, rmse, ccc, mae])

    df_results = pd.DataFrame(results)
    df_results.columns = ['method', 'model', 'functionals', 'kelm_c', 'kelm_kernel', 'RMSE', 'CCC', 'MAE']
    df_results.to_csv(f'C:/Users/mjkoe/Thesis/results/results_{pre}_KELM.csv')

    return


def get_functional_df_train_dev_test():
    # Get the features for this setting PDEM
    X_train_PDEM = pd.read_csv(
        f'C:/Users/mjkoe/Thesis/data/features/pdem_wav2vec/embed_train_{string_functionals}.csv',
        header=[0, 1], skipinitialspace=True, index_col=0)
    X_dev_PDEM = pd.read_csv(
        f'C:/Users/mjkoe/Thesis/data/features/pdem_wav2vec/embed_dev_{string_functionals}.csv',
        header=[0, 1], skipinitialspace=True, index_col=0)
    X_test_PDEM = pd.read_csv(
        f'C:/Users/mjkoe/Thesis/data/features/pdem_wav2vec/embed_test_{string_functionals}.csv',
        header=[0, 1], skipinitialspace=True, index_col=0)

    X_train_SBERT = pd.read_csv(
        f'C:/Users/mjkoe/Thesis/data/features/sbert_all-MiniLM-L12-v2/sbert_train_{string_functionals}.csv',
        header=[0, 1], skipinitialspace=True, index_col=0)
    X_dev_SBERT = pd.read_csv(
        f'C:/Users/mjkoe/Thesis/data/features/sbert_all-MiniLM-L12-v2/sbert_dev_{string_functionals}.csv',
        header=[0, 1], skipinitialspace=True, index_col=0)
    X_test_SBERT = pd.read_csv(
        f'C:/Users/mjkoe/Thesis/data/features/sbert_all-MiniLM-L12-v2/sbert_test_{string_functionals}.csv',
        header=[0, 1], skipinitialspace=True, index_col=0)

    X_train_VAD = pd.read_csv(
        f'C:/Users/mjkoe/Thesis/data/features/pdem_vad/vad_train_{string_functionals}.csv',
        header=[0, 1], skipinitialspace=True, index_col=0)
    X_dev_VAD = pd.read_csv(
        f'C:/Users/mjkoe/Thesis/data/features/pdem_vad/vad_dev_{string_functionals}.csv',
        header=[0, 1], skipinitialspace=True, index_col=0)
    X_test_VAD = pd.read_csv(
        f'C:/Users/mjkoe/Thesis/data/features/pdem_vad/vad_test_{string_functionals}.csv',
        header=[0, 1], skipinitialspace=True, index_col=0)

    return (X_train_SBERT, X_dev_SBERT, X_test_SBERT, X_train_PDEM, X_dev_PDEM, X_test_PDEM,
            X_train_VAD, X_dev_VAD, X_test_VAD)
