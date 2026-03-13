import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("../data/winequality.csv", sep=";")

# Features and labels
X = data.drop("quality", axis=1)
y = data["quality"]

# Convert to binary classification
y = (y >= 7).astype(int)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)

model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print("Model Accuracy:", accuracy)

# Save model
with open("../model/wine_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully.")
