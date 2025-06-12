import requests
import json

BASE_URL = "http://127.0.0.1:8000"
API_KEY = "my-secret-key-1234"  # The same key as in main.py
HEADERS = {"X-API-Key": API_KEY}

def test_get_root():
    """
    Tests the GET request to the root endpoint, which should now return HTML.
    """
    print("--- Testing GET request to / ---")
    try:
        response = requests.get(BASE_URL)
        response.raise_for_status()
        print(f"Status Code: {response.status_code}")
        # CORRECTED: Since this endpoint now serves an HTML file, we print a confirmation
        # instead of trying to parse it as JSON.
        print("Response Body: Received HTML content as expected.")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

def test_post_predict():
    """
    Tests the POST request to the /predict endpoint.
    """
    print("\n--- Testing POST request to /predict ---")
    
    high_income_data = {
        "age": 45, "workclass": "Private", "fnlwgt": 160000, "education": "Masters",
        "education-num": 14, "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial", "relationship": "Husband", "race": "White",
        "sex": "Male", "capital-gain": 5000, "capital-loss": 0,
        "hours-per-week": 50, "native-country": "United-States"
    }
    
    print("\nTesting with a valid API key:")
    try:
        response = requests.post(f"{BASE_URL}/predict", json=high_income_data, headers=HEADERS)
        response.raise_for_status()
        print(f"Status Code: {response.status_code}")
        print(f"Response JSON: {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        if e.response is not None:
            print(f"API Response: {e.response.json()}")

    print("\nTesting with an invalid API key:")
    try:
        invalid_headers = {"X-API-Key": "wrong-key"}
        response = requests.post(f"{BASE_URL}/predict", json=high_income_data, headers=invalid_headers)
        print(f"Status Code: {response.status_code}")
        print(f"Response JSON: {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    print("Testing API endpoints...")
    test_get_root()
    test_post_predict()