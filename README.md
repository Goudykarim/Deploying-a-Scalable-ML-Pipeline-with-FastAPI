# **Deploying a Scalable ML Pipeline with FastAPI**

Project Repository Link: https://github.com/Goudykarim/Deploying-a-Scalable-ML-Pipeline-with-FastAPI

(Note: Please ensure this link points to your public repository)

## **Project Overview**

This project demonstrates a complete end-to-end machine learning pipeline for a census income classification task. The primary goal is to predict whether an individual's annual income exceeds $50,000 based on 1994 US Census data.

The project encompasses data ingestion and cleaning, model training and evaluation, model slicing analysis, and deployment as a RESTful API using FastAPI. It also includes advanced features such as API authentication, a simple web-based frontend, and a robust Continuous Integration/Continuous Delivery (CI/CD) pipeline using GitHub Actions.

## **Key Features**

- **ML Model:** A `RandomForestClassifier` trained on the UCI Census Income dataset.
- **RESTful API:** A robust API built with FastAPI to serve model predictions, with automatic interactive documentation.
- **API Authentication:** The prediction endpoint is protected and requires a valid API key for access.
- **Interactive Frontend:** A user-friendly web interface built with HTML and vanilla JavaScript to interact with the API in real-time.
- **CI/CD Pipeline:** Automated workflows using GitHub Actions to lint, test, and build the application on every push to the main branch.
- **Code Coverage:** Integration with Codecov to monitor test coverage and ensure code quality.
- **Data Slicing Analysis:** The training pipeline generates a `slice_output.txt` report to evaluate model performance across different categorical features.

## **Installation & Setup**

To run this project locally, please follow these steps.

1. **Clone the Repository:**
    
    ```
    git clone https://github.com/Goudykarim/Deploying-a-Scalable-ML-Pipeline-with-FastAPI.git
    cd Deploying-a-Scalable-ML-Pipeline-with-FastAPI
    
    ```
    
2. **Create and Activate a Virtual Environment:**
    
    ```
    python3 -m venv .venv
    source .venv/bin/activate
    # On Windows, use: .venv\Scripts\activate
    
    ```
    
3. **Install Dependencies:**
    
    ```
    pip install -r requirements.txt
    
    ```
    

## **How to Run the Application**

### **1. Train the Model**

Before running the API, you must first train the model and generate the necessary artifacts. This only needs to be done once.

```
python train_model.py

```

This script will create the `model/` directory and save the trained `model.pkl`, `encoder.pkl`, and `lb.pkl` inside it.

### **2. Run the Unit Tests**

To verify the functionality of the core machine learning components and API endpoints:

```
python -m pytest test_ml.py -v

```

### **3. Launch the API Server**

Once the model is trained, you can launch the API server.

```
python -m uvicorn main:app --reload

```

The server will be available at `http://127.0.0.1:8000`.

### **4. Interact with the Application**

- **Frontend:** Open your web browser and navigate to **`http://127.0.0.1:8000`**. You will see the interactive web form.
- **API Docs:** For interactive API documentation (Swagger UI), navigate to **`http://127.0.0.1:8000/docs`**.
- **Scripted Interaction:** You can also test the API using the provided client script:
    
    ```
    python local_api.py
    
    ```
    

## **Deployment Strategy (Continuous Delivery)**

This project is configured for **Continuous Delivery (CD)** using **GitHub Actions**. The workflow defined in `.github/workflows/ci.yml` serves as the CD pipeline.

On every push to the `main` branch, the pipeline automatically executes a series of jobs that validate the application and deliver a new, stable version of the code to the repository. The key stages are:

1. **Build & Install:** The environment is set up and all dependencies are installed.
2. **Lint:** The code is checked for style and quality issues using `flake8`.
3. **Test:** The complete unit test suite is run using `pytest` to ensure all functionality is working as expected. This includes testing the live API endpoints locally within the runner.
4. **Generate** Production **Artifacts:** The `train_model.py` script is executed to produce the final, trained model artifacts (`.pkl` files). This simulates the build step that prepares the model for a production environment.

This automated process ensures that the `main` branch always represents a stable, tested, and "deployable" version of the application. A successful workflow run serves as evidence of a successful "deployment" to the repository's production-ready state.

## **Project Screenshots**

### **`unit_test.png` - Successful Unit Test Run**

### **`example.png` - API Documentation with Pydantic Example**

### **`live_post.png` - Live API POST Request from Script**

### **`live_get.png` - Live API GET Request to Root**

### **`continuous_deployment.png` - Successful CI/CD Pipeline Run**