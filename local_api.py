# local_api.py
import requests
import json

# Define the base URL of your locally running API
BASE_URL = "http://127.0.0.1:8000"

def test_get_root():
    """
    Tests the GET request to the root endpoint.
    """
    print("--- Testing GET request to / ---")
    try:
        response = requests.get(BASE_URL)
        response.raise_for_status()  # Raise an exception for bad status codes
        print(f"Status Code: {response.status_code}")
        print(f"Response JSON: {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

def test_post_predict():
    """
    Tests the POST request to the /predict endpoint.
    """
    print("\n--- Testing POST request to /predict ---")
    
    # Example data point that is likely to be '>50K'
    high_income_data = {
        "age": 45,
        "workclass": "Private",
        "fnlwgt": 160000, # CORRECTED: fnlgt -> fnlwgt
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 5000,
        "capital-loss": 0,
        "hours-per-week": 50,
        "native-country": "United-States"
    }

    # Example data point that is likely to be '<=50K'
    low_income_data = {
        "age": 25,
        "workclass": "Private",
        "fnlwgt": 250000, # CORRECTED: fnlgt -> fnlwgt
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Never-married",
        "occupation": "Other-service",
        "relationship": "Own-child",
        "race": "Black",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 35,
        "native-country": "United-States"
    }
    
    print("\nTesting with a high-income profile:")
    try:
        response_high = requests.post(f"{BASE_URL}/predict", json=high_income_data)
        response_high.raise_for_status()
        print(f"Status Code: {response_high.status_code}")
        print(f"Response JSON: {response_high.json()}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        # For debugging, print the detailed error from the API response
        if e.response is not None:
            print(f"API Response: {e.response.json()}")
        
    print("\nTesting with a low-income profile:")
    try:
        response_low = requests.post(f"{BASE_URL}/predict", json=low_income_data)
        response_low.raise_for_status()
        print(f"Status Code: {response_low.status_code}")
        print(f"Response JSON: {response_low.json()}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        if e.response is not None:
            print(f"API Response: {e.response.json()}")


if __name__ == "__main__":
    # Make sure your FastAPI server is running before executing this script
    # uvicorn main:app --reload
    test_get_root()
    test_post_predict()