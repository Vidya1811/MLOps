# Heart Disease Prediction Lab — MLflow Experiment Tracking

This lab demonstrates the full machine learning lifecycle using **MLflow**: data preprocessing, exploratory data analysis, model training with experiment tracking, hyperparameter tuning, model comparison, model registry, batch inference, and real-time serving.

**Adapted from** the [Wine Quality MLflow Lab2](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Experiment_Tracking_Labs/Mlflow_Labs/Lab2).

---

## Modifications from Original Lab

| Area | Original (Wine Quality) | Modified (Heart Disease) |
|------|------------------------|--------------------------|
| **Dataset** | UCI Wine Quality (red + white CSVs) | UCI Heart Disease (Cleveland), auto-downloaded |
| **Preprocessing** | Concatenation + `is_red` indicator variable | Binary target conversion (0–4 → 0/1) + missing value handling (`?` → NaN → drop) |
| **Feature Scaling** | None | `StandardScaler` fit on train, applied to val/test; embedded in model wrapper |
| **Models Trained** | 1 model: untuned Random Forest (10 trees) | 3 models: baseline RF, tuned RF via `GridSearchCV`, Gradient Boosting Classifier |
| **Hyperparameter Tuning** | None | `GridSearchCV` with 3-fold CV over `n_estimators`, `max_depth`, `min_samples_split` |
| **Metrics Logged** | AUC only | AUC, Precision, Recall, F1-score |
| **Artifacts Logged** | Model only | Model + confusion matrix PNG per run + feature importance chart + correlation heatmap |
| **Model Selection** | Manual (hardcoded run name lookup) | Automatic best-model selection by highest test AUC across all runs |
| **Model Registry** | Stage-based (`transition_model_version_stage` → "Production") | Alias-based (`set_registered_model_alias` → "champion") — uses the modern MLflow API |
| **Visualization** | `distplot` of quality + box plots | Count plot + box plots + correlation heatmap + horizontal bar chart of feature importances |
| **Model Wrapper** | Wraps model only | Wraps both model and scaler so production model accepts raw (unscaled) input |

---

## Prerequisites

- **Python** 3.8+ (tested on 3.9)
- **pip** package manager
- **Git** for cloning the repository

## Setup

```bash
# 1. Clone your fork
git clone https://github.com/<your-username>/MLOps.git
cd MLOps/Labs/Experiment_Tracking_Labs/Mlflow_Labs/Lab2

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install pandas numpy scikit-learn mlflow seaborn matplotlib cloudpickle

# 4. (Optional) Verify MLflow is installed
mlflow --version
```

---

## Dataset

**UCI Heart Disease — Cleveland subset**

| Property | Detail |
|----------|--------|
| **Source** | [UCI ML Repository — Heart Disease](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) |
| **Original Creators** | Hungarian Institute of Cardiology, University Hospital Zurich, University Hospital Basel, V.A. Medical Center |
| **Samples** | 303 total, 297 after dropping rows with missing values |
| **Features** | 13 clinical attributes |
| **Target** | Originally 0–4 (severity); converted to binary: 0 = no disease, 1 = disease present |
| **Class Balance** | 160 no disease (54%) / 137 disease (46%) after preprocessing |

### Feature Descriptions

| Feature | Description | Type |
|---------|-------------|------|
| `age` | Age in years | Continuous |
| `sex` | Sex (1 = male, 0 = female) | Binary |
| `cp` | Chest pain type (1–4) | Categorical |
| `trestbps` | Resting blood pressure (mm Hg) | Continuous |
| `chol` | Serum cholesterol (mg/dl) | Continuous |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) | Binary |
| `restecg` | Resting ECG results (0–2) | Categorical |
| `thalach` | Maximum heart rate achieved | Continuous |
| `exang` | Exercise-induced angina (1 = yes, 0 = no) | Binary |
| `oldpeak` | ST depression induced by exercise relative to rest | Continuous |
| `slope` | Slope of peak exercise ST segment (1–3) | Categorical |
| `ca` | Number of major vessels colored by fluoroscopy (0–3) | Discrete |
| `thal` | Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect) | Categorical |

The script **downloads the data automatically** from the UCI repository at runtime. No manual download is needed.

---

## Running the Lab

```bash
# Make sure your venv is activated, then:
python heart_disease_mlflow_lab.py
```

### Expected Output

The script prints progress for each step. A successful run ends with:

```
============================================================
Lab Complete!
============================================================
```

### Generated Files

After running, you'll find these files in your working directory:

| File | Description |
|------|-------------|
| `target_distribution.png` | Bar chart showing class balance (disease vs no disease) |
| `eda_boxplots.png` | Box plots of continuous features grouped by target |
| `correlation_heatmap.png` | Heatmap of pairwise feature correlations |
| `feature_importances.png` | Horizontal bar chart of Random Forest feature importances |
| `test_confusion_matrix.png` | Confusion matrix from each run (also logged to MLflow) |
| `mlruns/` | MLflow tracking directory containing all experiment data |

---

## Step-by-Step Walkthrough

### Steps 1–2: Import Libraries and Load Data

The Cleveland heart disease dataset is fetched directly from the UCI repository URL. If the download fails (e.g., no internet), it falls back to a local `data/heart.csv` file.

```python
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
data = pd.read_csv(url, names=column_names, na_values="?")
```

The dataset has 303 rows and 14 columns (13 features + 1 target).

### Step 3: Data Preprocessing

Three operations are performed:

1. **Binary target conversion**: The original target column has values 0–4 representing severity of heart disease. We convert this to binary — any value > 0 becomes 1 (disease present).
   ```python
   data["target"] = (data["target"] > 0).astype(int)
   ```
2. **Missing value handling**: The `ca` and `thal` columns contain `?` characters (loaded as NaN). We drop these 6 rows, leaving 297 samples.
3. **Column name cleaning**: Strip whitespace and replace spaces with underscores.

### Step 4: Data Visualization

A count plot of the target distribution is generated and saved. The dataset is reasonably balanced: 160 negative (54%) and 137 positive (46%).

### Step 5: Exploratory Data Analysis (EDA)

Box plots are created for the five continuous features (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`) grouped by the binary target. This helps identify which features show separation between disease and no-disease groups.

Key observations:
- **`thalach`** (max heart rate): Patients with heart disease tend to have lower max heart rates.
- **`oldpeak`** (ST depression): Higher values are associated with disease presence.
- **`age`**: Patients with heart disease tend to be older.

### Step 6: Correlation Heatmap

A full 14×14 correlation matrix is computed and plotted as a heatmap. This reveals multicollinearity between features and shows which features correlate most strongly with the target.

### Step 7: Data Splitting

The data is split into three sets:

| Set | Proportion | Samples | Purpose |
|-----|-----------|---------|---------|
| Train | 60% | 178 | Model fitting |
| Validation | 20% | 59 | Hyperparameter tuning (available for future use) |
| Test | 20% | 60 | Final evaluation |

`random_state=42` ensures reproducibility.

### Step 8: Feature Scaling

`StandardScaler` is fit on the training data only and applied to all three splits. This prevents data leakage from validation/test sets.

The scaler is embedded inside the `SklearnModelWrapper` class so the registered production model can accept raw (unscaled) input and handle scaling internally:

```python
class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def predict(self, context, model_input):
        scaled = self.scaler.transform(model_input)
        return self.model.predict_proba(scaled)[:, 1]
```

### Step 9: Baseline Model — Untuned Random Forest

A Random Forest with just 10 estimators and default hyperparameters serves as the baseline.

**Logged to MLflow:**
- Parameters: `model_type`, `n_estimators`, `scaling`
- Metrics: `test_auc`, `test_precision`, `test_recall`, `test_f1`
- Artifacts: confusion matrix PNG, the model itself

**Typical results:** AUC ≈ 0.94, F1 ≈ 0.89

### Step 10: Feature Importance Analysis

Feature importances from the baseline Random Forest are extracted and plotted. Top features typically include:
- `thalach` (max heart rate achieved)
- `ca` (number of major vessels)
- `thal` (thalassemia type)
- `age`

### Step 11: Tuned Random Forest (GridSearchCV)

`GridSearchCV` performs an exhaustive search over:

| Parameter | Values Tested |
|-----------|--------------|
| `n_estimators` | 50, 100, 200 |
| `max_depth` | 5, 10, None |
| `min_samples_split` | 2, 5 |

This is 3 × 3 × 2 = **18 parameter combinations**, each evaluated with 3-fold cross-validation (54 fits total), optimizing for AUC.

**Typical best params:** `{'max_depth': 5, 'min_samples_split': 5, 'n_estimators': 200}`

**Typical results:** AUC ≈ 0.97, F1 ≈ 0.87

### Step 12: Gradient Boosting Classifier

A `GradientBoostingClassifier` is trained as a second model type for comparison:
- `n_estimators=100`, `learning_rate=0.1`, `max_depth=5`

**Typical results:** AUC ≈ 0.92, F1 ≈ 0.84

### Step 13: Model Registration

The script automatically finds the best run by sorting all runs on `test_auc` in descending order:

```python
best_run = mlflow.search_runs(order_by=["metrics.test_auc DESC"]).iloc[0]
```

The best model is registered in the MLflow Model Registry under the name `heart_disease_prediction`.

### Step 14: Set Production Alias

Instead of the deprecated stage-based workflow (`transition_model_version_stage`), we use the modern **alias-based** approach:

```python
client.set_registered_model_alias(
    name=model_name,
    alias="champion",
    version=model_version.version,
)
```

This labels the best model version as `"champion"`, which can be loaded via `models:/{name}@champion`.

### Step 15: Champion Model Inference

The champion model is loaded from the registry and used for batch inference on the test set as a sanity check:

```python
prod_model = mlflow.pyfunc.load_model(f"models:/{model_name}@champion")
prod_preds = prod_model.predict(X_test)
```

The printed AUC should match the value logged during training.

### Step 16: Model Serving (Manual)

To serve the model as a REST API for real-time inference:

```bash
mlflow models serve -m "models:/heart_disease_prediction@champion" -h 0.0.0.0 -p 5001
```

Then send predictions from Python:

```python
import requests

url = "http://localhost:5001/invocations"
sample = {
    "dataframe_split": {
        "columns": ["age", "sex", "cp", "trestbps", "chol", "fbs",
                     "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"],
        "data": [[63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6]]
    }
}
response = requests.post(url, json=sample)
print(response.json())  # Returns probability of heart disease
```

---

## Viewing the MLflow UI

After running the lab, launch the tracking UI:

```bash
mlflow ui --port 5000
```

Open **http://localhost:5000** in your browser. You can:

1. **Compare runs**: Select all three runs and click "Compare" to see metrics side-by-side.
2. **View artifacts**: Click into any run to see logged confusion matrix images.
3. **Check parameters**: See which hyperparameters were used for each run.
4. **Model Registry**: Navigate to "Models" tab to see registered versions and aliases.

---

## Model Comparison Summary

| Model | AUC | Precision | Recall | F1 | Notes |
|-------|-----|-----------|--------|----|-------|
| Baseline RF (10 trees) | ~0.94 | ~0.88 | ~0.89 | ~0.89 | Quick baseline, no tuning |
| Tuned RF (GridSearchCV) | **~0.97** | ~0.85 | ~0.89 | ~0.87 | Best AUC, auto-registered as champion |
| Gradient Boosting | ~0.92 | ~0.84 | ~0.84 | ~0.84 | Alternative model type |

*Exact values vary slightly depending on the random state and data split.*

The tuned Random Forest achieves the highest AUC and is automatically selected and registered as the champion model.

---

## Project Structure

```
Lab2/
├── heart_disease_mlflow_lab.py    # Main lab script (all steps)
├── README.md                       # This documentation
├── target_distribution.png         # Generated: target class distribution
├── eda_boxplots.png               # Generated: EDA box plots by target
├── correlation_heatmap.png        # Generated: feature correlation heatmap
├── feature_importances.png        # Generated: RF feature importance bar chart
├── test_confusion_matrix.png      # Generated: confusion matrix (last run)
├── mlruns/                        # Generated: MLflow tracking data
│   ├── <experiment_id>/           # Experiment folder
│   │   ├── <run_id_1>/           # Baseline RF run
│   │   ├── <run_id_2>/           # Tuned RF run
│   │   └── <run_id_3>/           # Gradient Boosting run
│   └── models/                    # Model registry data
└── venv/                          # Virtual environment (not committed)
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` for any package | Run `pip install pandas numpy scikit-learn mlflow seaborn matplotlib cloudpickle` |
| `NotOpenSSLWarning` about LibreSSL | Harmless warning on macOS with older Python; can be ignored |
| `Registered model already exists` | Previous run created the model. This is fine — MLflow creates a new version |
| Model download fails from UCI | Check internet connection, or place `heart.csv` in a `data/` subdirectory |
| MLflow UI not loading | Make sure you're in the same directory where `mlruns/` was created |

---

## Key Takeaways

1. **Experiment Tracking**: MLflow logs parameters, metrics, and artifacts for every run, making model comparison straightforward and reproducible.
2. **Model Comparison**: Training multiple models (baseline RF, tuned RF, Gradient Boosting) and comparing them in the MLflow UI helps systematically identify the best approach rather than relying on guesswork.
3. **Hyperparameter Tuning**: `GridSearchCV` automates the search for optimal parameters, and logging the results to MLflow provides a permanent record of what was tried.
4. **Model Registry & Aliases**: Registering models and assigning aliases like `"champion"` provides a clean deployment workflow that separates model development from production serving.
5. **Reproducibility**: Logging scaling parameters, random seeds, conda environments, and all preprocessing steps ensures experiments can be exactly reproduced by anyone.
6. **End-to-End Pipeline**: The lab covers the entire ML lifecycle — from raw data to a served REST API — demonstrating how MLflow ties each stage together.