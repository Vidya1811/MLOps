# FastAPI ML Model Deployment (Modified Lab)

---

- **Video Explanation (Original Lab):** [FastAPI Lab](https://www.youtube.com/watch?v=KReburHqRIQ&list=PLcS4TrUUc53LeKBIyXAaERFKBJ3dvc9GZ&index=4)
- **Blog:** [FastAPI Lab-1](https://www.mlwithramin.com/blog/fastapi-lab1)

---

## Overview

In this lab, we demonstrate how to deploy a machine learning model as a REST API using **FastAPI** and **Uvicorn**.

### Technologies Used

- **FastAPI** — A modern, high-performance Python framework used to build APIs quickly using Python type hints.
- **Uvicorn** — An ASGI (Asynchronous Server Gateway Interface) web server used to run FastAPI applications.
- **Scikit-Learn** — A popular machine learning library used here to train a **Random Forest classifier**.

---

## Modifications from the Original Lab

The original lab used:

- **Dataset:** Iris dataset
- **Model:** Decision Tree Classifier
- **Task:** Multiclass classification of iris flowers

To satisfy the assignment requirement that the lab **must not be identical to the repository**, the following modifications were implemented:

| Component  | Original Lab          | Modified Implementation                  |
|------------|-----------------------|------------------------------------------|
| Dataset    | Iris dataset          | Wine Quality dataset                     |
| Model      | Decision Tree         | Random Forest (scikit-learn)             |
| Prediction | Single predict endpoint | Single + Batch prediction endpoints    |
| API        | Basic endpoints       | Additional model information endpoint    |
| Schema     | Iris feature schema   | Wine chemical features schema            |

These modifications demonstrate the ability to adapt the lab to **a new dataset, model, and API functionality**.

---

## Project Workflow

The pipeline implemented in this lab follows these steps:

1. Download the **Wine Quality dataset**
2. Train a **RandomForestClassifier** using scikit-learn
3. Save the trained model as a **pickle file**
4. Load the trained model inside a **FastAPI application**
5. Serve the model as an **ML inference API**

---

## Setting Up the Lab

### 1. Clone the Repository

```bash
git clone <your-forked-repo-link>
cd FastAPI_Labs/fastapi_lab1
```

### 2. Create a Virtual Environment

```bash
python3 -m venv fastapi_env
```

Activate it:

```bash
# Mac / Linux
source fastapi_env/bin/activate

# Windows
fastapi_env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If needed:

```bash
pip install fastapi uvicorn scikit-learn pandas numpy pydantic
```

---

## Dataset

The dataset used in this implementation is the **Wine Quality dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality).

### Download the dataset

```bash
mkdir -p data
curl -L https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv -o data/winequality.csv
```

### Dataset Description

The dataset contains chemical measurements of wines such as fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, sulfur dioxide levels, density, pH, sulphates, and alcohol content.

The goal is to predict wine quality. For this lab, the task was converted into **binary classification**:

- `quality >= 7` → **High quality wine (1)**
- `quality < 7` → **Low quality wine (0)**

---

## Project Structure

```
fastapi_lab1/
│
├── data/
│   └── winequality.csv
│
├── model/
│   └── wine_model.pkl
│
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── main.py
│   ├── predict.py
│   └── train.py
│
├── assets/
├── requirements.txt
└── README.md
```

| File         | Description                                  |
|--------------|----------------------------------------------|
| `train.py`   | Trains the Random Forest model               |
| `predict.py` | Loads the trained model and performs prediction |
| `data.py`    | Defines Pydantic request schemas             |
| `main.py`    | FastAPI application with API endpoints        |

---

## Training the Model

Move into the `src` directory:

```bash
cd src
```

Train the model:

```bash
python train.py
```

This script:

1. Loads the Wine Quality dataset
2. Splits data into training and testing sets
3. Trains a `RandomForestClassifier`
4. Evaluates model accuracy
5. Saves the trained model to `model/wine_model.pkl`

---

## Running the FastAPI Application

From the `src` directory, run:

```bash
uvicorn main:app --reload
```

The API server will start at:

```
http://127.0.0.1:8000
```

---

## API Documentation (Swagger UI)

FastAPI automatically generates interactive documentation. Open:

```
http://127.0.0.1:8000/docs
```

You will see all available endpoints and can test them directly.

---

## Available API Endpoints

### 1. Root Endpoint

```
GET /
```

Returns API status message.

**Example response:**

```json
{
  "message": "Wine Quality Prediction API"
}
```

---

### 2. Predict Wine Quality

```
POST /predict
```

Predicts the quality of a single wine sample.

**Example request:**

```json
{
  "fixed_acidity": 7.4,
  "volatile_acidity": 0.7,
  "citric_acid": 0,
  "residual_sugar": 1.9,
  "chlorides": 0.076,
  "free_sulfur_dioxide": 11,
  "total_sulfur_dioxide": 34,
  "density": 0.9978,
  "pH": 3.51,
  "sulphates": 0.56,
  "alcohol": 9.4
}
```

**Example response:**

```json
{
  "prediction": 0
}
```

| Value | Meaning            |
|-------|--------------------|
| 0     | Low quality wine   |
| 1     | High quality wine  |

---

### 3. Batch Prediction

```
POST /batch_predict
```

Allows prediction for multiple samples at once.

**Example request:**

```json
{
  "samples": [
    {
      "fixed_acidity": 7.4,
      "volatile_acidity": 0.7,
      "citric_acid": 0,
      "residual_sugar": 1.9,
      "chlorides": 0.076,
      "free_sulfur_dioxide": 11,
      "total_sulfur_dioxide": 34,
      "density": 0.9978,
      "pH": 3.51,
      "sulphates": 0.56,
      "alcohol": 9.4
    }
  ]
}
```

**Example response:**

```json
{
  "predictions": [0]
}
```

---

### 4. Model Information

```
GET /model_info
```

Returns metadata about the deployed model.

**Example response:**

```json
{
  "model": "RandomForestClassifier",
  "dataset": "Wine Quality Dataset",
  "task": "Wine quality classification"
}
```

---

## FastAPI Syntax Overview

Create FastAPI app:

```python
app = FastAPI()
```

Run server:

```bash
uvicorn main:app --reload
```

Where:

- `main` = file name (`main.py`)
- `app` = FastAPI instance
- `--reload` = auto-reload during development

---

## Pydantic Data Models

FastAPI uses **Pydantic** to validate request data.

**Example schema:**

```python
class WineData(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float
```

Pydantic automatically:

- Reads JSON request bodies
- Converts data types
- Validates input values
- Returns errors if inputs are invalid

---

## FastAPI Features Demonstrated

- **Request Handling** — FastAPI automatically reads JSON request bodies.
- **Data Validation** — Pydantic ensures correct data types and required fields.
- **Automatic API Documentation** — Swagger UI documentation is generated automatically.
- **Machine Learning Model Serving** — A trained ML model is deployed as a REST API.

---

## Conclusion

This lab demonstrates how to:

1. Train a machine learning model with scikit-learn
2. Serialize the model using pickle
3. Deploy the model using FastAPI
4. Serve predictions through REST API endpoints

The lab was extended from the original implementation by using a different dataset, model, and additional API endpoints, making the implementation more flexible and realistic for production ML systems.