import os
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import joblib

# --- Authentication Setup ---
api_key_header = APIKeyHeader(name="X-API-Key")
API_KEY = "my-secret-key-1234"

def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key == API_KEY:
        return api_key
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )
# --- End Authentication Setup ---


# Define paths
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.pkl")
LB_PATH = os.path.join(MODEL_DIR, "lb.pkl")

# Load artifacts
try:
    model, encoder, lb = joblib.load(MODEL_PATH), joblib.load(ENCODER_PATH), joblib.load(LB_PATH)
    print("Model and artifacts loaded successfully.")
except FileNotFoundError:
    print("Error: Model or artifacts not found.")
    model, encoder, lb = None, None, None

# Initialize FastAPI app
app = FastAPI(
    title="Census Income Prediction API",
    description="An API to predict whether income exceeds $50K/yr based on census data.",
    version="1.0.0"
)

# --- Mount Frontend ---
# This serves the static files (HTML, CSS, JS) from the 'frontend' directory
app.mount("/static", StaticFiles(directory="frontend"), name="static")


# --- API Data Model ---
class CensusData(BaseModel):
    age: int = Field(..., example=39)
    workclass: str = Field(..., example="State-gov")
    fnlwgt: int = Field(..., example=77516)
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


# --- API Endpoints ---
@app.get("/", response_class=FileResponse)
async def root():
    """Serves the main frontend page."""
    return "frontend/index.html"


@app.post("/predict", dependencies=[Depends(get_api_key)])
async def predict(data: CensusData):
    """Predicts the income category for the given census data."""
    if not all([model, encoder, lb]):
        raise HTTPException(status_code=503, detail="Model not loaded")

    input_df = pd.DataFrame([data.dict(by_alias=True)])
    
    from ml.data import process_data
    input_df['salary'] = '<=50K'

    X, _, _, _ = process_data(
        input_df,
        categorical_features=[
            "workclass", "education", "marital-status", "occupation",
            "relationship", "race", "sex", "native-country",
        ],
        label="salary", training=False, encoder=encoder, lb=lb
    )

    prediction_raw = model.predict(X)
    prediction = lb.inverse_transform(prediction_raw)[0]

    return {"prediction": prediction}