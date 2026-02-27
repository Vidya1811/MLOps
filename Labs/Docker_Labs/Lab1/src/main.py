import argparse
import json
import os
from datetime import datetime

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def parse_args():
    p = argparse.ArgumentParser(
        description="Train a breast cancer classifier and save artifacts."
    )
    p.add_argument(
        "--output-dir", default="artifacts", help="Directory to write model + metrics"
    )
    p.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    p.add_argument("--random-state", type=int, default=42, help="Random seed")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load DIFFERENT dataset (not Iris)
    data = load_breast_cancer()
    X, y = data.data, data.target

    # 2) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # 3) DIFFERENT model & better practice: pipeline with scaling
    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, solver="liblinear")),
        ]
    )
    clf.fit(X_train, y_train)

    # 4) Evaluate
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    metrics = {
        "accuracy": acc,
        "f1": f1,
        "test_size": args.test_size,
        "random_state": args.random_state,
    }

    report = classification_report(y_test, preds, output_dict=True)

    # 5) Save artifacts (model + metrics + metadata)
    model_path = os.path.join(args.output_dir, "model.joblib")
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    metadata_path = os.path.join(args.output_dir, "metadata.json")

    joblib.dump(clf, model_path)

    with open(metrics_path, "w") as f:
        json.dump({"metrics": metrics, "classification_report": report}, f, indent=2)

    with open(metadata_path, "w") as f:
        json.dump(
            {
                "created_at": datetime.utcnow().isoformat() + "Z",
                "dataset": "sklearn.datasets.load_breast_cancer",
                "model": "StandardScaler + LogisticRegression(liblinear)",
                "artifact_paths": {
                    "model": model_path,
                    "metrics": metrics_path,
                },
            },
            f,
            indent=2,
        )

    print("✅ Training complete")
    print(f"Model saved to:   {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Accuracy={acc:.4f}  F1={f1:.4f}")
