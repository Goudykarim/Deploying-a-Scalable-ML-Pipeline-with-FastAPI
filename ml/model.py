# ml/model.py
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import joblib
import os


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model : sklearn.ensemble.RandomForestClassifier
        Trained machine learning model.
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the model on calculated metrics and returns them.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    f1 : float
    """
    f1 = f1_score(y, preds, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, f1


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : sklearn.ensemble.RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data to predict on.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def save_model(model, encoder, lb, model_path, encoder_path, lb_path):
    """
    Saves the trained model, encoder, and label binarizer to disk.
    
    Inputs
    ------
    model: Trained model object
    encoder: Trained OneHotEncoder object
    lb: Trained LabelBinarizer object
    model_path: str
    encoder_path: str
    lb_path: str
    """
    joblib.dump(model, model_path)
    joblib.dump(encoder, encoder_path)
    joblib.dump(lb, lb_path)


def load_model(model_path, encoder_path, lb_path):
    """
    Loads a trained model, encoder, and label binarizer from disk.

    Inputs
    ------
    model_path: str
    encoder_path: str
    lb_path: str

    Returns
    -------
    model: Loaded model object
    encoder: Loaded encoder object
    lb: Loaded LabelBinarizer object
    """
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    lb = joblib.load(lb_path)
    return model, encoder, lb


def performance_on_categorical_slice(model, data, categorical_feature, label, encoder, lb):
    """
    Computes model metrics on slices of the data for a given categorical feature.
    """
    from ml.data import process_data
    
    output_lines = []

    for cls in data[categorical_feature].unique():
        temp_df = data[data[categorical_feature] == cls]
        
        X_slice, y_slice, _, _ = process_data(
            temp_df,
            categorical_features=[
                "workclass", "education", "marital-status", "occupation",
                "relationship", "race", "sex", "native-country"
            ],
            label=label,
            training=False,
            encoder=encoder,
            lb=lb
        )

        if len(X_slice) == 0:
            continue

        preds = inference(model, X_slice)
        precision, recall, f1 = compute_model_metrics(y_slice, preds)

        line = f"'{cls}' - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
        output_lines.append(line)
        
    return "\n".join(output_lines)