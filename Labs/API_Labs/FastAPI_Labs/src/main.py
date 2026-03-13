from fastapi import FastAPI
from data import WineData, WineBatch
from predict import predict_wine

app = FastAPI()


@app.get("/")
def home():
    return {"message": "Wine Quality Prediction API"}


@app.post("/predict")
def predict(data: WineData):

    prediction = predict_wine(data)

    return {"prediction": prediction}


@app.post("/batch_predict")
def batch_predict(batch: WineBatch):

    predictions = []

    for sample in batch.samples:
        pred = predict_wine(sample)
        predictions.append(pred)

    return {"predictions": predictions}


@app.get("/model_info")
def model_info():

    return {
        "model": "RandomForestClassifier",
        "dataset": "Wine Quality Dataset",
        "task": "Wine quality classification",
    }
