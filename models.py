import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, normalize
from evaluation.kernel_elm import ELM


def sym_model(kelm_c: float, kelm_kernel: str, x_train: pd.DataFrame, y_train: np.array, x_input: pd.DataFrame,
              weight_kelm=False, classify=False, gender_vec=None, gender_aware=False):
    """
    :param y_train is a matrix with n*8 symptom values per participant in the range [0-3]
    """
    # Scale and normalize the SBERT embeddings
    scaler = StandardScaler().fit(x_train)
    X_train_scaled = normalize(scaler.transform(x_train))
    X_input_scaled = normalize(scaler.transform(x_input))

    # adding a bias term to the features
    bias_train = np.ones(shape=(X_train_scaled.shape[0], 1))
    bias_input = np.ones(shape=(X_input_scaled.shape[0], 1))
    X_train_scaled = np.hstack([bias_train, X_train_scaled])
    X_input_scaled = np.hstack([bias_input, X_input_scaled])

    elm = ELM(c=kelm_c, kernel=kelm_kernel, is_classification=classify, weighted=weight_kelm, gender_vec=gender_vec, gender_aware=gender_aware)
    elm.fit(X_train_scaled, y_train, gender_vec=gender_vec)
    y_pred = elm.predict(X_input_scaled)

    if classify:
        y_pred = np.argmax(y_pred,axis=0)

    # # Sanitizing symptoms
    y_pred[y_pred < 0] = 0
    y_pred[y_pred > 3] = 3

    # Rounding symptoms to nearest integer
    unrounded_symptoms = y_pred
    rounded_symptoms = np.rint(y_pred)
    pred_severity = np.rint(np.sum(y_pred, axis=1))  # sum the symptom prediction scores to total severity

    return unrounded_symptoms, rounded_symptoms, pred_severity


def sev_model(kelm_c: float, kelm_kernel: str, x_train: pd.DataFrame, y_train: np.array, x_input: pd.DataFrame,
              weight_kelm=False, classify=False, gender_vec=None, gender_aware=False):
    """
    :param y_train is a vector of phq8 severity scores for each participant
    """
    # Scale and normalize the SBERT embeddings
    scaler = StandardScaler().fit(x_train)
    X_train_scaled = normalize(scaler.transform(x_train))
    X_input_scaled = normalize(scaler.transform(x_input))

    # adding a bias term to the features
    bias_train = np.ones(shape=(X_train_scaled.shape[0], 1))
    bias_input = np.ones(shape=(X_input_scaled.shape[0], 1))
    X_train_scaled = np.hstack([bias_train, X_train_scaled])
    X_input_scaled = np.hstack([bias_input, X_input_scaled])

    elm = ELM(c=kelm_c, kernel=kelm_kernel, is_classification=classify, weighted=weight_kelm, gender_vec=gender_vec, gender_aware=gender_aware)
    elm.fit(X_train_scaled, y_train, gender_vec=gender_vec)
    pred_severity = elm.predict(X_input_scaled)

    if classify:
        pred_severity = np.argmax(pred_severity,axis=1)

    # # Sanitizing symptoms
    pred_severity[pred_severity < 0] = 0
    pred_severity[pred_severity > 24] = 24

    return np.round(pred_severity)


def bin_model(kelm_c: float, kelm_kernel: str, x_train: pd.DataFrame, y_train: np.array, x_input: pd.DataFrame,
              weight_kelm=False, classify=False, gender_vec=None, gender_aware=False):
    """
    :param y_train is a vector of binary depression scores for each participant
    """
    # Scale and normalize the SBERT embeddings
    scaler = StandardScaler().fit(x_train)
    X_train_scaled = normalize(scaler.transform(x_train))
    X_input_scaled = normalize(scaler.transform(x_input))

    # adding a bias term to the features
    bias_train = np.ones(shape=(X_train_scaled.shape[0], 1))
    bias_input = np.ones(shape=(X_input_scaled.shape[0], 1))
    X_train_scaled = np.hstack([bias_train, X_train_scaled])
    X_input_scaled = np.hstack([bias_input, X_input_scaled])

    elm = ELM(c=kelm_c, kernel=kelm_kernel, is_classification=classify, weighted=weight_kelm, gender_vec=gender_vec,gender_aware=gender_aware)
    elm.fit(X_train_scaled, y_train, gender_vec=gender_vec)
    pred_binary = elm.predict(X_input_scaled)

    if classify:
        pred_binary = np.argmax(pred_binary,axis=1)

    # # Sanitizing symptoms
    pred_binary[pred_binary < 0] = 0
    pred_binary[pred_binary > 1] = 1

    return np.rint(pred_binary)
