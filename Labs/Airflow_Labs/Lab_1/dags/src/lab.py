import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
import pickle
import os
import base64
import json
from datetime import datetime


# ---- Paths (container-safe + local-safe) ----
BASE_DIR = os.path.dirname(__file__)  # dags/src
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))   # dags/data
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "model")) # dags/model
os.makedirs(MODEL_DIR, exist_ok=True)


def _b64_pickle(obj) -> str:
    """Pickle -> base64 string (XCom JSON-safe)."""
    return base64.b64encode(pickle.dumps(obj)).decode("ascii")


def _unb64_pickle(s: str):
    """base64 string -> unpickle."""
    return pickle.loads(base64.b64decode(s))


def load_data():
    """
    Loads Mall Customers CSV, cleans it, serializes it, and returns JSON-safe base64.
    CHANGE vs starter:
      - Uses mall_customers.csv (your downloaded dataset) instead of file.csv
      - Drops ID column
      - Encodes Gender
      - Selects numeric features automatically
    """
    csv_path = os.path.join(DATA_DIR, "mall_customers.csv")  # rename your file to this
    if not os.path.exists(csv_path):
        # fallback to original name if you didn't rename
        alt_path = os.path.join(DATA_DIR, "Mall_Customers.csv")
        if os.path.exists(alt_path):
            csv_path = alt_path
        else:
            raise FileNotFoundError(
                f"Could not find dataset. Expected {os.path.join(DATA_DIR,'mall_customers.csv')} "
                f"or {os.path.join(DATA_DIR,'Mall_Customers.csv')}"
            )

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]  # handle stray spaces

    # Drop ID-like column if present
    for id_col in ["CustomerID", "CustomerId", "customer_id", "ID", "id"]:
        if id_col in df.columns:
            df = df.drop(columns=[id_col])

    # Encode Gender if present (custom change)
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
        # If unexpected strings exist, coerce to numeric safely
        df["Gender"] = pd.to_numeric(df["Gender"], errors="coerce")

    # Keep only numeric columns for clustering
    df = df.select_dtypes(include="number")

    return _b64_pickle(df)


def data_preprocessing(data_b64: str):
    """
    Deserializes base64-encoded pickled DataFrame, performs preprocessing,
    and returns base64-encoded pickled numpy array of features.

    CHANGE vs starter:
      - No longer hard-codes BALANCE/PURCHASES/CREDIT_LIMIT
      - Works for Mall Customers (and any numeric CSV)
      - Simple NaN handling
    """
    df = _unb64_pickle(data_b64)

    # Fill missing numeric values with column means (safer than dropna for small datasets)
    df = df.fillna(df.mean(numeric_only=True))

    scaler = MinMaxScaler()
    X = scaler.fit_transform(df.values)

    # Save scaler artifact (nice + reproducible)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    return _b64_pickle(X)


def build_save_model(data_b64: str, filename: str, k_max: int = 12):
    """
    Builds KMeans models for k=1..k_max, records SSE, saves the "best-k" model using elbow.
    Returns SSE list (JSON-serializable).

    CHANGE vs starter:
      - Uses elbow to decide best k and saves THAT model (instead of saving k=49)
      - k_max reduced to something reasonable for this dataset (default 12)
      - Writes SSE artifact
    """
    X = _unb64_pickle(data_b64)

    # guard
    n_samples = X.shape[0]
    k_max = int(min(max(2, k_max), max(2, n_samples - 1)))

    kmeans_kwargs = {"init": "k-means++", "n_init": 10, "max_iter": 300, "random_state": 42}
    sse = []
    models = {}

    for k in range(1, k_max + 1):
        km = KMeans(n_clusters=k, **kmeans_kwargs)
        km.fit(X)
        sse.append(float(km.inertia_))
        models[k] = km

    # Choose best k via elbow
    ks = list(range(1, k_max + 1))
    kl = KneeLocator(ks, sse, curve="convex", direction="decreasing")
    best_k = kl.elbow if kl.elbow is not None else min(3, k_max)

    best_model = models[int(best_k)]
    output_path = os.path.join(MODEL_DIR, filename)
    with open(output_path, "wb") as f:
        pickle.dump(best_model, f)

    # Save SSE + best_k artifact
    with open(os.path.join(MODEL_DIR, "sse_and_k.json"), "w") as f:
        json.dump(
            {"k_max": k_max, "sse": sse, "best_k": int(best_k), "saved_model": filename},
            f,
            indent=2,
        )

    return sse


def load_model_elbow(filename: str, sse: list):
    """
    Loads the saved model and reports best k. Produces a prediction on the dataset itself
    (cluster labels) and returns a JSON-safe summary dict.

    CHANGE vs starter:
      - No dependency on test.csv (so it's self-contained)
      - Writes metrics.json artifact
      - Returns useful summary
    """
    model_path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = pickle.load(open(model_path, "rb"))

    # Recompute elbow_k just for reporting (based on SSE passed through XCom)
    ks = list(range(1, len(sse) + 1))
    kl = KneeLocator(ks, sse, curve="convex", direction="decreasing")
    elbow_k = kl.elbow if kl.elbow is not None else min(3, len(sse))

    # Load the dataset again to generate cluster labels (self-contained)
    # (Uses same logic as load_data for locating file)
    csv_path = os.path.join(DATA_DIR, "mall_customers.csv")
    if not os.path.exists(csv_path):
        alt_path = os.path.join(DATA_DIR, "Mall_Customers.csv")
        csv_path = alt_path if os.path.exists(alt_path) else csv_path

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    for id_col in ["CustomerID", "CustomerId", "customer_id", "ID", "id"]:
        if id_col in df.columns:
            df = df.drop(columns=[id_col])

    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
        df["Gender"] = pd.to_numeric(df["Gender"], errors="coerce")

    df = df.select_dtypes(include="number").fillna(df.mean(numeric_only=True))
    X = MinMaxScaler().fit_transform(df.values)  # quick scale for labeling

    labels = model.predict(X)

    result = {
        "optimal_k_elbow": int(elbow_k),
        "model_n_clusters": int(getattr(model, "n_clusters", -1)),
        "num_rows_labeled": int(len(labels)),
        "label_counts": {str(k): int(v) for k, v in pd.Series(labels).value_counts().items()},
        "computed_at": datetime.utcnow().isoformat() + "Z",
    }

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result