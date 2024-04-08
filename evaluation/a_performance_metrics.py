import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, mean_absolute_error


def tp_fp_tn_fn(df: pd.DataFrame) -> tuple:
    """
    :param df: dataframe with 3 columns: gender, binary true labels and binary predictions
    :return: true positive, false positive, true negative, false negative
    """
    TP = np.sum((df['binary_true'] == 1) & (df['binary_pred'] == 1))
    FP = np.sum((df['binary_true'] == 0) & (df['binary_pred'] == 1))
    TN = np.sum((df['binary_true'] == 0) & (df['binary_pred'] == 0))
    FN = np.sum((df['binary_true'] == 1) & (df['binary_pred'] == 0))
    return TP, FP, TN, FN


def binary_metrics(gender_vec: pd.Series, binary_trues: pd.Series, binary_preds: list) -> tuple:
    """
    :param gender_vec:      vector with binary gender labels
    :param binary_trues:    vector with binary true depression labels
    :param binary_preds:    vector with binary depression predictions
    :return:                binary performance metrics
    """
    df_info = pd.DataFrame({'gender': gender_vec.values,
                            'binary_true': binary_trues.values,
                            'binary_pred': binary_preds})

    gender_0 = df_info.query('gender == 0')
    gender_1 = df_info.query('gender == 1')

    TP, FP, TN, FN = tp_fp_tn_fn(df_info)
    TP_0, FP_0, TN_0, FN_0 = tp_fp_tn_fn(gender_0)
    TP_1, FP_1, TN_1, FN_1 = tp_fp_tn_fn(gender_1)

    # acc = (TP + TN) / (TP + TN + FP + FN)
    # acc_0 = (TP_0 + TN_0) / (TP_0 + TN_0 + FP_0 + FN_0)
    # acc_1 = (TP_1 + TN_1) / (TP_1 + TN_1 + FP_1 + FN_1)

    sensitivity = TP / (TP + FN)  # recall / TPR
    sensitivity_0 = TP_0 / (TP_0 + FN_0)
    sensitivity_1 = TP_1 / (TP_1 + FN_1)

    # specificity = TN / (TN + FP)  # TNR
    specificity_0 = TN_0 / (TN_0 + FP_0)
    specificity_1 = TN_1 / (TN_1 + FP_1)

    precision = TP / (TP + FP)  # Positive predictive value
    precision_0 = TP_0 / (TP_0 + FP_0)
    precision_1 = TP_1 / (TP_1 + FP_1)

    # FNR = 1 - sensitivity
    # FNR_0 = 1 - sensitivity_0
    # FNR_1 = 1 - sensitivity_1

    # FPR = 1 - specificity
    FPR_0 = 1 - specificity_0
    FPR_1 = 1 - specificity_1

    F1 = (2 * precision * sensitivity) / (precision + sensitivity)
    F1_0 = (2 * precision_0 * sensitivity_0) / (precision_0 + sensitivity_0)
    F1_1 = (2 * precision_1 * sensitivity_1) / (precision_1 + sensitivity_1)

    prob_pos_0 = (TP_0 + FP_0) / (TP_0 + TN_0 + FP_0 + FN_0)
    prob_pos_1 = (TP_1 + FP_1) / (TP_0 + TN_0 + FP_0 + FN_0)

    SP_measure = prob_pos_1 / prob_pos_0
    Eopp_measure = sensitivity_1 / sensitivity_0
    FPR_measure = FPR_1 / FPR_0
    Eodd_measure = min(FPR_measure, Eopp_measure)

    # print('FPR female', FPR_0, 'FPR male', FPR_1)
    return SP_measure, Eopp_measure, FPR_measure, Eodd_measure, F1, F1_0, F1_1


def concordance_correlation_coefficient(y_true: pd.Series, y_pred: pd.Series):
    """
    :param y_true: Series with true severity
    :param y_pred: Series with predicted severity
    :return: the CCC value
    """
    # Pearson product-moment correlation coefficients
    cor = np.corrcoef(y_true, y_pred)[0][1]
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)

    # Calculate CCC
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

    return numerator / denominator


def non_binary_metrics(trues: pd.Series, preds: pd.Series) -> tuple:
    """
    :param trues:   true severity scores
    :param preds:   predicted severity scores
    :return:        non-binary metrics
    """
    rmse = np.round(root_mean_squared_error(trues, preds), decimals=4)
    ccc = np.round(concordance_correlation_coefficient(trues, preds), decimals=4)
    mae = np.round(mean_absolute_error(trues, preds), decimals=4)
    return rmse, ccc, mae


def performance_measures(df: pd.DataFrame) -> None:
    """
    :param df:  dataframe with columns gender, true_binary, true_severity, pred_severity
    :return:    prints performance measures, returns none
    """
    print('Performance metrics:')

    if 'pred_severity' in df.columns:
        print('\t\tOverall\t Female\t Male\t| CCC\t\t| MAE')

        binary_predictions = list(map(int, df['pred_severity'] >= 10))
        rmse, ccc, mae = non_binary_metrics(df['true_severity'], df['pred_severity'])

        rmse_g0 = np.round(root_mean_squared_error(df.query('gender == 0')['true_severity'],
                                                   df.query('gender == 0')['pred_severity']), decimals=4)
        rmse_g1 = np.round(root_mean_squared_error(df.query('gender == 1')['true_severity'],
                                                   df.query('gender == 1')['pred_severity']), decimals=4)
        print(f'RMSE:\t{rmse}\t {rmse_g0}\t {rmse_g1}\t| {ccc}\t| {mae}')
    else:
        print('\t\tOverall\t Female\t Male')
        binary_predictions = df['pred_binary']

    SP_measure, Eopp_ratio, FPR_ratio, Eodd_ratio, F1, F1_0, F1_1 = binary_metrics(gender_vec=df['gender'],
                                                                                   binary_trues=df['true_binary'],
                                                                                   binary_preds=binary_predictions)

    print(f'F1:\t\t{np.round(F1, 4)}\t {np.round(F1_0, 4)}\t {np.round(F1_1, 4)}')
    print(f'Equal opportunity ratio M/F:    {np.round(Eopp_ratio, 4)}')
    print(f'False Positive Rate ratio M/F:  {np.round(FPR_ratio, 4)}')
    # print(f'Equalized odds ratio M/F:       {np.round(Eodd_ratio, 4)}\n')

    return
