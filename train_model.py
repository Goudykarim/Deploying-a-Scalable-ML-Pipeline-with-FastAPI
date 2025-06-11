# train_model.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics, save_model, performance_on_categorical_slice

# Define constants
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.pkl")
LB_PATH = os.path.join(MODEL_DIR, "lb.pkl") # Path for the label binarizer
SLICE_OUTPUT_PATH = "slice_output.txt"

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Define categorical features
CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

def run_training_pipeline():
    """
    Executes the full model training and evaluation pipeline.
    """
    print("1. Fetching data from UCI repository...")
    census_income = fetch_ucirepo(id=20)
    X = census_income.data.features
    y = census_income.data.targets

    # Combine features and target into a single dataframe
    data = X.copy()
    data['salary'] = y['income']

    # --- Robust Data Cleaning ---
    # Strip whitespace from all object columns
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].str.strip()

    # Explicitly clean the target column to remove periods and ensure consistency
    data['salary'] = data['salary'].str.replace('.', '', regex=False)

    # Handle missing values represented as '?'
    data.replace("?", pd.NA, inplace=True)
    data.dropna(inplace=True)
    # --- End of Cleaning ---

    print("2. Splitting data into training and testing sets...")
    train, test = train_test_split(data, test_size=0.20, random_state=42, stratify=data['salary'])

    print("3. Processing training data...")
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=CAT_FEATURES, label="salary", training=True
    )

    print("4. Training the model...")
    model = train_model(X_train, y_train)
    print("Model training complete.")

    print(f"5. Saving model, encoder, and label binarizer to '{MODEL_DIR}/'...")
    save_model(model, encoder, lb, MODEL_PATH, ENCODER_PATH, LB_PATH)
    print("Artifacts saved.")

    print("6. Processing test data and running inference...")
    X_test, y_test, _, _ = process_data(
        test, categorical_features=CAT_FEATURES, label="salary", training=False, encoder=encoder, lb=lb
    )
    
    preds = inference(model, X_test)

    print("7. Computing overall model metrics on the test set...")
    precision, recall, f1 = compute_model_metrics(y_test, preds)
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1 Score: {f1:.4f}")
    
    print("8. Computing and saving model performance on data slices...")
    with open(SLICE_OUTPUT_PATH, 'w') as f:
        for feature in CAT_FEATURES:
            f.write(f"--- Performance for feature: {feature} ---\n")
            slice_performance = performance_on_categorical_slice(
                model, test, feature, "salary", encoder, lb
            )
            f.write(slice_performance)
            f.write("\n\n")
    print(f"Slice performance metrics saved to '{SLICE_OUTPUT_PATH}'.")


if __name__ == "__main__":
    run_training_pipeline()