"""
Heart Disease Prediction Lab — MLflow Experiment Tracking
=========================================================
Adapted from the Wine Quality MLflow Lab2.
Dataset: UCI Heart Disease (Cleveland)
Task: Binary classification — predict presence of heart disease.

Modifications from original lab:
  - Different dataset (heart disease instead of wine quality)
  - Added StandardScaler preprocessing
  - Added XGBoost as a second model for comparison
  - Added precision, recall, F1 alongside AUC
  - Added confusion matrix logging as an artifact
  - Added hyperparameter tuning with GridSearchCV
"""

# ============================================================
# Step 1: Import Libraries
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
import cloudpickle
import time
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# Step 2: Load and Explore Data
# ============================================================
print("=" * 60)
print("Step 2: Loading Data")
print("=" * 60)

# UCI Heart Disease — Cleveland dataset
column_names = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# Try loading from URL; fall back to local file
try:
    data = pd.read_csv(url, names=column_names, na_values="?")
    print("Loaded data from UCI repository.")
except Exception:
    data = pd.read_csv("data/heart.csv")
    print("Loaded data from local file.")

print(f"Dataset shape: {data.shape}")
print(data.head())

# ============================================================
# Step 3: Data Preprocessing
# ============================================================
print("\n" + "=" * 60)
print("Step 3: Preprocessing")
print("=" * 60)

# The original target has values 0-4; convert to binary (0 = no disease, 1 = disease)
data["target"] = (data["target"] > 0).astype(int)

# Handle missing values — drop rows with NaN (only a few in 'ca' and 'thal')
print(f"Missing values before cleaning:\n{data.isna().sum()}")
data = data.dropna()
print(f"Dataset shape after dropping NaN: {data.shape}")

# Clean column names (replace spaces if any)
data.rename(columns=lambda x: x.strip().replace(" ", "_"), inplace=True)

print(f"\nTarget distribution:\n{data['target'].value_counts()}")

# ============================================================
# Step 4: Data Visualization
# ============================================================
print("\n" + "=" * 60)
print("Step 4: Visualization")
print("=" * 60)

# Distribution of target
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x="target", data=data, ax=ax)
ax.set_title("Heart Disease Distribution (0=No, 1=Yes)")
ax.set_xlabel("Target")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig("target_distribution.png")
plt.close()
print("Saved target_distribution.png")

# ============================================================
# Step 5: Exploratory Data Analysis — Box Plots
# ============================================================
continuous_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]

fig, axes = plt.subplots(1, len(continuous_features), figsize=(20, 5))
for i, col in enumerate(continuous_features):
    sns.boxplot(x="target", y=col, data=data, ax=axes[i])
    axes[i].set_title(col)
plt.suptitle("Feature Distributions by Heart Disease Presence", y=1.02)
plt.tight_layout()
plt.savefig("eda_boxplots.png")
plt.close()
print("Saved eda_boxplots.png")

# ============================================================
# Step 6: Correlation Heatmap (additional EDA)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()
print("Saved correlation_heatmap.png")

# ============================================================
# Step 7: Data Splitting
# ============================================================
print("\n" + "=" * 60)
print("Step 7: Splitting Data")
print("=" * 60)

X = data.drop("target", axis=1)
y = data["target"]

X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.6, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_rem, y_rem, test_size=0.5, random_state=42
)

print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

# ============================================================
# Step 8: Feature Scaling (NEW — not in original lab)
# ============================================================
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train), columns=X.columns, index=X_train.index
)
X_val_scaled = pd.DataFrame(
    scaler.transform(X_val), columns=X.columns, index=X_val.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test), columns=X.columns, index=X_test.index
)
print("Applied StandardScaler to features.")


# ============================================================
# Helper: Custom Model Wrapper
# ============================================================
class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
    """Wraps a sklearn classifier to return probability of positive class."""

    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def predict(self, context, model_input):
        scaled = self.scaler.transform(model_input)
        return self.model.predict_proba(scaled)[:, 1]


# ============================================================
# Helper: Log metrics & confusion matrix
# ============================================================
def log_classification_metrics(y_true, y_prob, prefix=""):
    """Log AUC, precision, recall, F1 to MLflow."""
    y_pred = (y_prob >= 0.5).astype(int)
    auc = roc_auc_score(y_true, y_prob)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    mlflow.log_metric(f"{prefix}auc", auc)
    mlflow.log_metric(f"{prefix}precision", prec)
    mlflow.log_metric(f"{prefix}recall", rec)
    mlflow.log_metric(f"{prefix}f1", f1)

    # Save confusion matrix as artifact
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Disease", "Disease"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax)
    ax.set_title(f"{prefix}Confusion Matrix")
    fname = f"{prefix}confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    mlflow.log_artifact(fname)

    return {"auc": auc, "precision": prec, "recall": rec, "f1": f1}


# ============================================================
# Step 9: Baseline Model — Untuned Random Forest
# ============================================================
print("\n" + "=" * 60)
print("Step 9: Baseline Random Forest")
print("=" * 60)

mlflow.set_experiment("heart-disease-prediction")

with mlflow.start_run(run_name="untuned_random_forest"):
    n_estimators = 10
    model_rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model_rf.fit(X_train_scaled, y_train)

    preds_test = model_rf.predict_proba(X_test_scaled)[:, 1]

    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("scaling", "StandardScaler")

    metrics = log_classification_metrics(y_test, preds_test, prefix="test_")
    print(f"Baseline RF — AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")

    # Log model
    wrapped = SklearnModelWrapper(model_rf, scaler)
    signature = infer_signature(X_train, wrapped.predict(None, X_train))

    conda_env = _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=[
            f"cloudpickle=={cloudpickle.__version__}",
            f"scikit-learn=={sklearn.__version__}",
        ],
        additional_conda_channels=None,
    )
    mlflow.pyfunc.log_model(
        "random_forest_model",
        python_model=wrapped,
        conda_env=conda_env,
        signature=signature,
    )

# ============================================================
# Step 10: Feature Importance
# ============================================================
print("\n" + "=" * 60)
print("Step 10: Feature Importance")
print("=" * 60)

feat_imp = pd.DataFrame(
    model_rf.feature_importances_, index=X_train.columns, columns=["importance"]
).sort_values("importance", ascending=False)
print(feat_imp)

fig, ax = plt.subplots(figsize=(10, 6))
feat_imp.plot.barh(ax=ax)
ax.set_title("Random Forest Feature Importances")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importances.png")
plt.close()
print("Saved feature_importances.png")

# ============================================================
# Step 11: Tuned Random Forest with GridSearchCV (NEW)
# ============================================================
print("\n" + "=" * 60)
print("Step 11: Tuned Random Forest (GridSearchCV)")
print("=" * 60)

with mlflow.start_run(run_name="tuned_random_forest"):
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5],
    }
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=3,
        scoring="roc_auc",
        n_jobs=-1,
    )
    grid_search.fit(X_train_scaled, y_train)
    best_rf = grid_search.best_estimator_

    preds_test = best_rf.predict_proba(X_test_scaled)[:, 1]

    mlflow.log_param("model_type", "RandomForest_Tuned")
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_param("scaling", "StandardScaler")

    metrics = log_classification_metrics(y_test, preds_test, prefix="test_")
    print(f"Tuned RF — Best params: {grid_search.best_params_}")
    print(f"Tuned RF — AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")

    wrapped_tuned = SklearnModelWrapper(best_rf, scaler)
    signature = infer_signature(X_train, wrapped_tuned.predict(None, X_train))
    mlflow.pyfunc.log_model(
        artifact_path="tuned_rf_model",
        python_model=wrapped_tuned,
        conda_env=conda_env,
        signature=signature,
    )

# ============================================================
# Step 12: Gradient Boosting Model (NEW — second model comparison)
# ============================================================
print("\n" + "=" * 60)
print("Step 12: Gradient Boosting Classifier")
print("=" * 60)

with mlflow.start_run(run_name="gradient_boosting"):
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
    )
    gb_model.fit(X_train_scaled, y_train)

    preds_test = gb_model.predict_proba(X_test_scaled)[:, 1]

    mlflow.log_param("model_type", "GradientBoosting")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("scaling", "StandardScaler")

    metrics = log_classification_metrics(y_test, preds_test, prefix="test_")
    print(f"GradientBoosting — AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")

    wrapped_gb = SklearnModelWrapper(gb_model, scaler)
    signature = infer_signature(X_train, wrapped_gb.predict(None, X_train))
    mlflow.pyfunc.log_model(
        artifact_path="gradient_boosting_model",
        python_model=wrapped_gb,
        conda_env=conda_env,
        signature=signature,
    )

# ============================================================
# Step 13: Register Best Model
# ============================================================
print("\n" + "=" * 60)
print("Step 13: Model Registration")
print("=" * 60)

# Find the best run by AUC
best_run = mlflow.search_runs(order_by=["metrics.test_auc DESC"]).iloc[0]
best_run_id = best_run.run_id
best_run_name = best_run["tags.mlflow.runName"]
print(f"Best run: {best_run_name} (AUC: {best_run['metrics.test_auc']:.4f})")

# Determine artifact path based on run name
if "gradient_boosting" in best_run_name:
    artifact_path = "gradient_boosting_model"
elif "tuned" in best_run_name:
    artifact_path = "tuned_rf_model"
else:
    artifact_path = "random_forest_model"

model_name = "heart_disease_prediction"
model_version = mlflow.register_model(
    f"runs:/{best_run_id}/{artifact_path}", model_name
)
print(f"Registered model '{model_name}' version {model_version.version}")
time.sleep(10)

# ============================================================
# Step 14: Set Model Alias to "champion"
# ============================================================
print("\n" + "=" * 60)
print("Step 14: Set Production Alias")
print("=" * 60)

from mlflow.tracking import MlflowClient

client = MlflowClient()
# Use aliases instead of the deprecated stage-based workflow
client.set_registered_model_alias(
    name=model_name,
    alias="champion",
    version=model_version.version,
)
print(f"Model version {model_version.version} aliased as 'champion'.")

# ============================================================
# Step 15: Load Champion Model & Evaluate
# ============================================================
print("\n" + "=" * 60)
print("Step 15: Champion Model Inference")
print("=" * 60)

prod_model = mlflow.pyfunc.load_model(f"models:/{model_name}@champion")
prod_preds = prod_model.predict(X_test)
prod_auc = roc_auc_score(y_test, prod_preds)
print(f"Champion model AUC on test set: {prod_auc:.4f}")

# ============================================================
# Step 16: Serve Model (instructions)
# ============================================================
print("\n" + "=" * 60)
print("Step 16: Model Serving")
print("=" * 60)
print(
    """
To serve the model for real-time inference, run:

    mlflow models serve -m "models:/heart_disease_prediction@champion" -h 0.0.0.0 -p 5001

Then send requests:

    import requests, json
    url = 'http://localhost:5001/invocations'
    data_dict = {"dataframe_split": X_test.to_dict(orient='split')}
    response = requests.post(url, json=data_dict)
    print(response.json())
"""
)

print("\n" + "=" * 60)
print("Lab Complete!")
print("=" * 60)
