# test_ml.py
import os
import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics

# Define a fixture to load data once for all tests
@pytest.fixture(scope="session")
def data():
    """Fixture to load and clean the census data for testing."""
    try:
        census_income = fetch_ucirepo(id=20)
        X = census_income.data.features
        y = census_income.data.targets
        df = X.copy()
        df['salary'] = y['income']

        # --- Robust Data Cleaning (Mirrors train_model.py) ---
        # Strip whitespace from all object columns
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.strip()

        # Explicitly clean the target column to remove periods
        df['salary'] = df['salary'].str.replace('.', '', regex=False)

        # Handle missing values represented as '?'
        df.replace("?", pd.NA, inplace=True)
        df.dropna(inplace=True)
        # --- End of Cleaning ---

        return df
    except Exception as e:
        pytest.fail(f"Dataset fetching and cleaning failed: {e}")

@pytest.fixture(scope="session")
def processed_data(data):
    """Fixture to process data for tests."""
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    train, _ = train_test_split(data, test_size=0.20, random_state=42)
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    return X_train, y_train, encoder, lb

def test_data_loading(data):
    """Test that the data loads correctly and has the expected shape."""
    assert data is not None
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    expected_cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'salary']
    for col in expected_cols:
        assert col in data.columns


def test_model_training(processed_data):
    """Test that the model training function returns a fitted model."""
    X_train, y_train, _, _ = processed_data
    model = train_model(X_train, y_train)
    assert hasattr(model, "feature_importances_")
    assert model.n_features_in_ == X_train.shape[1]

def test_inference_and_metrics(processed_data):
    """Test the inference and metrics computation."""
    X_train, y_train, _, _ = processed_data
    model = train_model(X_train, y_train)
    
    preds = inference(model, X_train)
    assert preds is not None
    assert len(preds) == len(y_train)

    precision, recall, f1 = compute_model_metrics(y_train, preds)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(f1, float)
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= f1 <= 1.0