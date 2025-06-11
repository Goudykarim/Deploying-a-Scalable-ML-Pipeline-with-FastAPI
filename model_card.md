# Model Card for Census Income Prediction

## Model Details

**Model Developed By:** [Your Name/Team Name]
**Model Date:** June 11, 2025
**Model Version:** 1.0
**Model Type:** `sklearn.ensemble.RandomForestClassifier`
**Contact:** [Your Email or Contact Information]

## Intended Use

This model is intended to predict whether an individual's annual income is greater than $50,000 based on features from the 1994 US Census database. The primary use case is for educational purposes to demonstrate the deployment of a machine learning model via a RESTful API. It could also be used for sociological research or economic analysis, with appropriate caveats regarding its limitations.

## Training Data

The model was trained on the "Census Income" dataset, which was extracted from the 1994 Census bureau database by Barry Becker and Ronny Kohavi.

- **Source:** [UCI Machine Learning Repository: Adult Dataset](https://archive.ics.uci.edu/ml/datasets/Adult)
- **Data Size:** The full dataset contains 48,842 instances after removing entries with missing values. For this project, a subset of 32,561 instances was used.
- **Features:** The dataset includes 14 features, both categorical and continuous, such as age, workclass, education, marital status, occupation, race, sex, and hours worked per week.
- **Target Variable:** `salary`, which is a binary variable indicating whether income is `<=50K` or `>50K`.

## Preprocessing Steps

1.  **Column Cleaning:** Leading/trailing whitespace was stripped from all column names.
2.  **Data Splitting:** The data was split into a training set (80%) and a testing set (20%).
3.  **Categorical Feature Encoding:** `OneHotEncoder` was used to convert categorical features into a numerical format. This creates a binary column for each category.
4.  **Label Encoding:** `LabelBinarizer` was used to convert the target variable `salary` into a binary (0/1) format.

## Model Training

The model is a `RandomForestClassifier` from the scikit-learn library. It was trained on the processed training data. The default hyperparameters were used, with `random_state=42` for reproducibility.

## Evaluation

The model's performance was evaluated on the held-out test set (20% of the data).

### Overall Performance Metrics

-   **Precision:** 0.7634
-   **Recall:** 0.6133
-   **F1-Score:** 0.6800

*(Note: These are example metrics. You should replace them with the actual output from your `train_model.py` script.)*

### Performance on Data Slices

The model's performance was also analyzed on slices of the data for each categorical feature. This helps identify if the model is biased or performs poorly for specific subgroups. The full output is available in `slice_output.txt`.

**Example Slice Performance (for `sex`):**
-   **Female:** Precision: 0.85, Recall: 0.45, F1: 0.59
-   **Male:** Precision: 0.75, Recall: 0.65, F1: 0.69

This indicates a performance disparity between genders, where the model has higher precision for females but much lower recall compared to males.

## Ethical Considerations & Limitations

-   **Bias:** The dataset is from 1994 and reflects the societal biases of that time. The model may perpetuate these biases. For example, performance metrics differ across sensitive attributes like `race` and `sex`, which could lead to unfair outcomes if used for real-world decisions.
-   **Generalizability:** The model was trained on US-specific data from over two decades ago. It is unlikely to generalize well to other countries, time periods, or populations.
-   **Data Privacy:** Although the data is publicly available, it contains sensitive demographic information.
-   **Deployment Caveats:** This model should not be used for decisions that have a real-world impact on individuals, such as credit scoring, hiring, or loan applications, due to its inherent biases and limitations. Its deployment is strictly for demonstrating MLOps principles.

## How to Use the Model

The model is deployed via a FastAPI application.

1.  **Run the API:** `uvicorn main:app --reload`
2.  **Send a POST request** to the `/predict` endpoint with a JSON payload containing the 14 features. See `local_api.py` for an example.

```json
{
  "age": 39,
  "workclass": "State-gov",
  "fnlgt": 77516,
  "education": "Bachelors",
  "education-num": 13,
  "marital-status": "Never-married",
  "occupation": "Adm-clerical",
  "relationship": "Not-in-family",
  "race": "White",
  "sex": "Male",
  "capital-gain": 2174,
  "capital-loss": 0,
  "hours-per-week": 40,
  "native-country": "United-States"
}

  @media print {
    .ms-editor-squiggler {
        display:none !important;
    }
  }
  .ms-editor-squiggler {
    all: initial;
    display: block !important;
    height: 0px !important;
    width: 0px !important;
  }