import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def process_data(
    X, categorical_features=[], label=None, training=True, encoder=None, lb=None
):
    """ Process the data used in the machine learning pipeline.

    Processes the data into features and labels for model training and inference.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label.
    categorical_features: list[str]
        List of column names to be treated as categorical features.
    label : str
        Name of the label column in `X`.
    training : bool
        Indicator for whether the processing is for training or inference.
    encoder : sklearn.preprocessing.OneHotEncoder
        Trained OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing.LabelBinarizer
        Trained LabelBinarizer, only used if training=False.

    Returns
    -------
    X_processed : np.array
        Processed feature data.
    y : np.array
        Processed label data.
    encoder : sklearn.preprocessing.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the input encoder.
    lb : sklearn.preprocessing.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the input lb.
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    # This is the robust fix:
    # Explicitly select continuous features by finding columns that are NOT categorical.
    continuous_features = [col for col in X.columns if col not in categorical_features]
    
    X_continuous = X[continuous_features]
    X_categorical = X[categorical_features]

    if training:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        except AttributeError:
            # This handles cases where y is not available (e.g., live prediction)
            pass

    # Ensure continuous data is a numpy array before concatenation
    X_continuous = X_continuous.to_numpy()

    X_processed = np.concatenate([X_continuous, X_categorical], axis=1)
    
    return X_processed, y, encoder, lb
