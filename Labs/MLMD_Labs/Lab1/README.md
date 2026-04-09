# MLMD Lab 1 — ML Metadata Pipeline for Heart Disease Classification

## Overview

This lab demonstrates how to use [ML Metadata (MLMD)](https://www.tensorflow.org/tfx/guide/mlmd) independently of TFX to manually track a complete ML pipeline. Starting from the original course walkthrough (which only covered data validation on the Chicago Taxi dataset), this lab has been significantly extended with a different dataset, additional pipeline stages, and persistent storage.

### Key Modifications from Original Lab

| Aspect | Original Lab | Modified Lab |
|--------|-------------|--------------|
| **Dataset** | Chicago Taxi (10K+ rows, 18 features) | UCI Heart Disease (297 rows, 13 features + target) |
| **Pipeline stages** | 1 (Data Validation only) | 4 (Data Validation, Anomaly Detection, Model Training, Model Evaluation) |
| **Artifact types** | 3 (DataSet, Schema, statistics) | 6 (DataSet, Schema, statistics, Anomalies, Model, ModelEvaluation) |
| **Execution types** | 1 (Data Validation) | 4 (Data Validation, Anomaly Detection, Model Training, Model Evaluation) |
| **Storage backend** | In-memory fake database | Persistent SQLite database (`metadata/mlmd.sqlite`) |
| **ML model** | None | RandomForest classifier (scikit-learn) |
| **Evaluation metrics** | None | Accuracy, F1 score, Precision, Recall |
| **Anomaly detection** | None | TFDV schema validation on eval data |

---

## Dataset: UCI Heart Disease (Cleveland)

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- **Samples**: 303 total, 297 after cleaning (rows with `?` values dropped)
- **Target**: Binary classification — 0 (no heart disease) vs 1 (heart disease present)
- **Features** (13):

| Feature | Description | Type |
|---------|-------------|------|
| `age` | Age in years | INT |
| `sex` | Sex (1 = male, 0 = female) | INT |
| `cp` | Chest pain type (1-4) | INT |
| `trestbps` | Resting blood pressure (mm Hg) | INT |
| `chol` | Serum cholesterol (mg/dl) | INT |
| `fbs` | Fasting blood sugar > 120 mg/dl | INT |
| `restecg` | Resting ECG results (0-2) | INT |
| `thalach` | Maximum heart rate achieved | INT |
| `exang` | Exercise-induced angina (1 = yes) | INT |
| `oldpeak` | ST depression induced by exercise | FLOAT |
| `slope` | Slope of peak exercise ST segment | INT |
| `ca` | Number of major vessels (0-3) | INT |
| `thal` | Thalassemia (3 = normal, 6 = fixed, 7 = reversible) | INT |

### Data Splits

| Split | Rows | Purpose |
|-------|------|---------|
| Train (60%) | 178 | Schema inference, model training |
| Eval (20%) | 59 | Anomaly detection, model evaluation |
| Serving (20%) | 60 | Inference simulation |

All splits are generated with `random_state=42` for reproducibility. The notebook downloads the data at runtime from the UCI repository, with a fallback to the pre-generated local CSV files.

---

## Pipeline Architecture

The notebook implements a 4-stage ML pipeline, with each stage fully tracked through MLMD:

```
                          ┌──────────────────┐
                          │  Dataset (train)  │
                          └────────┬─────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼               │
          ┌─────────────┐  ┌─────────────┐        │
          │    Data      │  │    Model    │        │
          │  Validation  │  │  Training   │        │
          └──────┬───────┘  └──────┬──────┘        │
                 │                 │                │
                 ▼                 ▼                │
          ┌──────────┐     ┌──────────┐            │
          │  Schema  │     │  Model   │            │
          └────┬─────┘     └────┬─────┘            │
               │                │                  │
               ▼                │       ┌──────────┘
        ┌─────────────┐        │       │
        │   Anomaly   │◄───────┼───────┘
        │  Detection  │        │   Dataset (eval)
        └──────┬──────┘        │
               │               ▼
               ▼        ┌─────────────┐
        ┌───────────┐   │    Model    │◄── Dataset (eval)
        │ Anomalies │   │ Evaluation  │
        └───────────┘   └──────┬──────┘
                               │
                               ▼
                      ┌─────────────────┐
                      │ ModelEvaluation  │
                      │ (metrics)        │
                      └─────────────────┘
```

All artifacts and executions are grouped under a single **Experiment Context** (`Heart Disease Pipeline`).

---

## Pipeline Stages in Detail

### Stage 1: Data Validation

Uses [TensorFlow Data Validation (TFDV)](https://www.tensorflow.org/tfx/data_validation/get_started) to generate statistics from the training data and infer a schema.

- **Input**: `DataSet` artifact (train split)
- **Output**: `Schema` artifact (`schema.pbtxt`)
- **TFDV functions**: `generate_statistics_from_csv()`, `infer_schema()`

### Stage 2: Anomaly Detection

Validates the eval dataset against the inferred schema to detect data drift, missing features, or unexpected values.

- **Inputs**: `Schema` artifact + `DataSet` artifact (eval split)
- **Output**: `Anomalies` artifact (`anomalies.pbtxt`) with `num_anomalies` count and `description`
- **TFDV functions**: `generate_statistics_from_csv()`, `validate_statistics()`

### Stage 3: Model Training

Trains a `RandomForestClassifier` (100 estimators, `random_state=42`) on the Heart Disease training data and saves the model as a pickle file.

- **Input**: `DataSet` artifact (train split)
- **Output**: `Model` artifact (`model/model.pkl`) with `framework=scikit-learn`

### Stage 4: Model Evaluation

Evaluates the trained model on the held-out eval dataset, computing four classification metrics.

- **Inputs**: `Model` artifact + `DataSet` artifact (eval split)
- **Output**: `ModelEvaluation` artifact (`model/eval_metrics.json`) with metrics stored as MLMD DOUBLE properties:
  - `accuracy`
  - `f1_score`
  - `precision`
  - `recall`

---

## MLMD Data Model

### Artifact Types (6)

| Type | Properties | Description |
|------|-----------|-------------|
| `statistics` | name (STRING), split (STRING), version (STRING) | Data statistics |
| `DataSet` | name (STRING), split (STRING), version (INT) | Input datasets |
| `Schema` | name (STRING), version (INT) | Inferred data schemas |
| `Anomalies` | name (STRING), num_anomalies (INT), description (STRING) | Anomaly detection results |
| `Model` | name (STRING), version (INT), framework (STRING) | Trained ML models |
| `ModelEvaluation` | name (STRING), accuracy (DOUBLE), f1_score (DOUBLE), precision (DOUBLE), recall (DOUBLE) | Evaluation metrics |

### Execution Types (4)

| Type | Properties | Description |
|------|-----------|-------------|
| `Data Validation` | state (STRING) | Schema inference via TFDV |
| `Anomaly Detection` | state (STRING) | Eval data validation against schema |
| `Model Training` | state (STRING) | RandomForest model training |
| `Model Evaluation` | state (STRING) | Model evaluation on eval split |

### Context

| Type | Instance | Note |
|------|----------|------|
| `Experiment` | `Heart Disease Pipeline` | Groups all 4 attributions + 4 associations |

### Lineage Tracking

The notebook demonstrates full provenance queries:
- **Schema lineage**: Schema -> Data Validation execution -> training dataset
- **Anomaly lineage**: Anomalies -> Anomaly Detection execution -> schema + eval dataset
- **Model lineage**: Model -> Model Training execution -> training dataset
- **Evaluation lineage**: ModelEvaluation -> Model Evaluation execution -> model + eval dataset
- **Pipeline summary**: All artifacts and executions in the experiment context

---

## Persistent Storage (SQLite)

Unlike the original lab which used an in-memory fake database, this lab uses a **SQLite-backed metadata store** at `metadata/mlmd.sqlite`. This means:

- Metadata survives after the notebook kernel shuts down
- You can query the store in a separate script or notebook session
- The database file can be inspected with any SQLite browser

The notebook removes and recreates the database on each full run to ensure reproducibility. To accumulate metadata across runs, remove the `os.remove()` line in the storage setup cell.

---

## Project Structure

```
Labs/MLMD_Labs/Lab1/
├── C2_W3_Lab_1_MLMetadata.ipynb   # Main notebook (69 cells)
├── schema.pbtxt                    # Pre-generated Heart Disease schema
├── README.md                       # This file
├── data/
│   ├── train/
│   │   └── data.csv               # 178 training samples
│   ├── eval/
│   │   └── data.csv               # 59 evaluation samples
│   └── serving/
│       └── data.csv               # 60 serving samples
├── img/
│   └── mlmd_overview.png           # MLMD architecture diagram
├── model/                          # Generated at runtime
│   ├── model.pkl                   # Trained RandomForest model
│   └── eval_metrics.json           # Evaluation metrics (JSON)
├── metadata/                       # Generated at runtime
│   └── mlmd.sqlite                 # Persistent MLMD database
└── anomalies.pbtxt                 # Generated at runtime — anomaly report
```

---

## Requirements

- Python 3.9+
- TensorFlow 2.x
- TensorFlow Data Validation (TFDV)
- ML Metadata (`ml-metadata`)
- scikit-learn
- pandas

Install dependencies:

```bash
pip install tensorflow tensorflow-data-validation ml-metadata scikit-learn pandas
```

---

## How to Run

1. Open `C2_W3_Lab_1_MLMetadata.ipynb` in Jupyter Notebook or JupyterLab
2. Run all cells sequentially (Kernel -> Restart & Run All)
3. The notebook will:
   - Download the Heart Disease dataset (or use local CSVs if offline)
   - Create a SQLite metadata store at `metadata/mlmd.sqlite`
   - Run all 4 pipeline stages with full MLMD tracking
   - Demonstrate lineage queries tracing artifacts back through the pipeline
4. After running, the SQLite database persists at `metadata/mlmd.sqlite` for future queries

---

## References

- [ML Metadata Documentation](https://www.tensorflow.org/tfx/guide/mlmd)
- [MLMD API Reference](https://www.tensorflow.org/tfx/ml_metadata/api_docs/python/mlmd/MetadataStore)
- [TFDV Documentation](https://www.tensorflow.org/tfx/data_validation/get_started)
- [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- [MetadataStore Protocol Buffer](https://github.com/google/ml-metadata/blob/master/ml_metadata/proto/metadata_store.proto)
