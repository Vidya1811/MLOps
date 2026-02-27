Docker Lab 1 (Updated for submission)

This lab trains a machine learning model inside a Docker container and saves the trained model + evaluation outputs as artifacts on the host machine via a mounted volume.

What I changed 

Compared to the original lab implementation, I made the following changes:

ML / code changes
	•	Dataset changed from Iris to Breast Cancer Wisconsin dataset (sklearn.datasets.load_breast_cancer)
	•	Model changed from RandomForest to a Pipeline: StandardScaler + LogisticRegression
	•	Added evaluation metrics: Accuracy, F1-score, and a full classification report (stored in JSON)
	•	Saves all outputs into an artifacts/ folder:
	•	artifacts/model.joblib
	•	artifacts/metrics.json
	•	artifacts/metadata.json
	•	Added CLI args for reproducibility and experimentation:
	•	--output-dir (default: artifacts)
	•	--test-size (default: 0.2)
	•	--random-state (default: 42)

Docker-specific changes
	•	Multi-stage Docker build:
	•	Builder stage uses pip wheel to build dependency wheels
	•	Runtime stage installs only the built wheels (*.whl) for a cleaner final image
	•	Uses python:3.10-slim for a smaller base image
	•	Runs container as a non-root user (appuser) for better security
	•	Added ENTRYPOINT + CMD so you can pass CLI arguments directly to the container:
	•	Default behavior runs training and writes artifacts to artifacts/
	•	You can override parameters like --test-size 0.3

⸻

Project structure

Lab1/
  ReadMe.md
  dockerfile
  src/
    main.py
    requirements.txt

Note: The Dockerfile is named dockerfile (lowercase). Commands below use -f dockerfile.

⸻

Build the Docker image

From inside the Lab1/ directory:

docker build -t docker-lab1-custom:v1 -f dockerfile .


⸻

Run the container (recommended: save artifacts to host)

Default run (writes to artifacts/)

rm -rf artifacts && mkdir -p artifacts
docker run --rm \
  -v "$(pwd)/artifacts:/app/artifacts" \
  docker-lab1-custom:v1

Run with custom parameters (example: larger test split)

rm -rf artifacts && mkdir -p artifacts
docker run --rm \
  -v "$(pwd)/artifacts:/app/artifacts" \
  docker-lab1-custom:v1 \
  --test-size 0.3

Run with custom output directory (still mounted)

rm -rf artifacts && mkdir -p artifacts
docker run --rm \
  -v "$(pwd)/artifacts:/app/artifacts" \
  docker-lab1-custom:v1 \
  --output-dir artifacts


⸻

Verify outputs

After running, confirm the artifact files were created:

ls -la artifacts

Expected files:
	•	model.joblib (trained model pipeline)
	•	metrics.json (accuracy, f1, classification report)
	•	metadata.json (dataset/model info + timestamp)

(Optional) Validate JSON formatting:

python -m json.tool artifacts/metrics.json | head
python -m json.tool artifacts/metadata.json | head


⸻

Example results (sample runs)

Default run:

✅ Training complete
Model saved to:   artifacts/model.joblib
Metrics saved to: artifacts/metrics.json
Accuracy=0.9825  F1=0.9861

Run with --test-size 0.3:

✅ Training complete
Model saved to:   artifacts/model.joblib
Metrics saved to: artifacts/metrics.json
Accuracy=0.9883  F1=0.9907


⸻

Optional: Export the image as a tar file

docker save docker-lab1-custom:v1 > docker-lab1-custom.tar


