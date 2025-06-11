# main.py
import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib

# Define the path to the model and encoder
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.pkl")
LB_PATH = os.path.join(MODEL_DIR, "lb.pkl") # Path for the label binarizer

# Load the trained model and artifacts at startup
try:
    model, encoder, lb = joblib.load(MODEL_PATH), joblib.load(ENCODER_PATH), joblib.load(LB_PATH)
    print("Model and artifacts loaded successfully.")
except FileNotFoundError:
    print("Error: Model or artifacts not found. Please run train_model.py first.")
    model, encoder, lb = None, None, None

# Initialize the FastAPI app
app = FastAPI(
    title="Census Income Prediction API",
    description="An API to predict whether income exceeds $50K/yr based on census data.",
    version="1.0.0"
)

# Define the input data model using Pydantic
class CensusData(BaseModel):
    age: int = Field(..., example=39)
    workclass: str = Field(..., example="State-gov")
    fnlwgt: int = Field(..., example=77516) # CORRECTED: fnlgt -> fnlwgt
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., alias="education-num", example=13)
    marital_status: str = Field(..., alias="marital-status", example="Never-married")
    occupation: str = Field(..., alias="occupation", example="Adm-clerical")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., alias="capital-gain", example=2174)
    capital_loss: int = Field(..., alias="capital-loss", example=0)
    hours_per_week: int = Field(..., alias="hours-per-week", example=40)
    native_country: str = Field(..., alias="native-country", example="United-States")

    class Config:
        allow_population_by_field_name = True


@app.get("/")
async def root():
    """ Welcome message for the API root. """
    return {"message": "Welcome to the Census Income Prediction API!"}


@app.post("/predict")
async def predict(data: CensusData):
    """ Predicts the income category for the given census data. """
    if not all([model, encoder, lb]):
        return {"error": "Model not loaded. Please ensure training is complete."}

    input_df = pd.DataFrame([data.dict(by_alias=True)])
    
    from ml.data import process_data

    # Add a dummy salary column as it's expected by process_data
    input_df['salary'] = '<=50K'

    X, _, _, _ = process_data(
        input_df,
        categorical_features=[
            "workclass", "education", "marital-status", "occupation",
            "relationship", "race", "sex", "native-country",
        ],
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )

    prediction_raw = model.predict(X)
    
    # Use the inverse_transform of the loaded label binarizer
    prediction = lb.inverse_transform(prediction_raw)[0]

    return {"prediction": prediction}