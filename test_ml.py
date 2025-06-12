# test_ml.py
import os
import pytest
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics

# --- Fixtures for Data ---
@pytest.fixture(scope="session")
def data():
    """Fixture to load and clean the census data for testing."""
    try:
        census_income = fetch_ucirepo(id=20)
        X = census_income.data.features
        y = census_income.data.targets
        df = X.copy()
        df['salary'] = y['income']
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.strip()
        df['salary'] = df['salary'].str.replace('.', '', regex=False)
        df.replace("?", pd.NA, inplace=True)
        df.dropna(inplace=True)
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

# --- Core ML Tests (3 tests) ---

def test_data_loading(data):
    """Test 1: That the data loads correctly and has expected columns."""
    assert data is not None
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    expected_cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'salary']
    for col in expected_cols:
        assert col in data.columns

def test_model_training(processed_data):
    """Test 2: That the model training function returns a fitted model."""
    X_train, y_train, _, _ = processed_data
    model = train_model(X_train, y_train)
    assert hasattr(model, "feature_importances_")
    assert model.n_features_in_ == X_train.shape[1]

def test_inference_and_metrics(processed_data):
    """Test 3: That inference and metrics computation return correct types and ranges."""
    X_train, y_train, _, _ = processed_data
    model = train_model(X_train, y_train)
    preds = inference(model, X_train)
    precision, recall, f1 = compute_model_metrics(y_train, preds)
    assert isinstance(precision, float) and 0.0 <= precision <= 1.0
    assert isinstance(recall, float) and 0.0 <= recall <= 1.0
    assert isinstance(f1, float) and 0.0 <= f1 <= 1.0

# --- Additional Tests (3 more tests) ---

def test_process_data_output_shape(data):
    """Test 4: That process_data produces output with the correct number of columns after one-hot encoding."""
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    train, _ = train_test_split(data, test_size=0.20, random_state=42)
    X_train_raw, y_train_raw, encoder, _ = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    
    num_continuous = len(train.columns) - len(cat_features) - 1 
    num_one_hot = sum(len(cats) for cats in encoder.categories_)
    
    expected_cols = num_continuous + num_one_hot
    assert X_train_raw.shape[1] == expected_cols

def test_api_get_root_live(live_server_url):
    """Test 5: That the live API root endpoint returns HTML."""
    try:
        r = requests.get(live_server_url)
        assert r.status_code == 200
        assert "text/html" in r.headers['Content-Type']
        assert "<!DOCTYPE html>" in r.text
    except requests.exceptions.ConnectionError as e:
        pytest.fail(f"Could not connect to the server at {live_server_url}. Is it running?")


def test_api_post_predict_no_auth(live_server_url):
    """Test 6: That the prediction endpoint correctly returns an auth error without an API key."""
    high_income_data = {
        "age": 45, "workclass": "Private", "fnlwgt": 160000, "education": "Masters",
        "education-num": 14, "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial", "relationship": "Husband", "race": "White",
        "sex": "Male", "capital-gain": 5000, "capital-loss": 0,
        "hours-per-week": 50, "native-country": "United-States"
    }
    try:
        r = requests.post(f"{live_server_url}/predict", json=high_income_data)
        # FINAL CORRECTION: Expecting 401, which is what the code explicitly returns.
        assert r.status_code == 401
    except requests.exceptions.ConnectionError as e:
        pytest.fail(f"Could not connect to the server at {live_server_url}.")

@pytest.fixture
def live_server_url():
    """Fixture to provide the live server URL for tests."""
    return "http://127.0.0.1:8000"