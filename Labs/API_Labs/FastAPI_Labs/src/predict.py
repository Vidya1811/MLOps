import pickle
import numpy as np

with open("../model/wine_model.pkl", "rb") as f:
    model = pickle.load(f)


def predict_wine(data):

    features = np.array(
        [
            data.fixed_acidity,
            data.volatile_acidity,
            data.citric_acid,
            data.residual_sugar,
            data.chlorides,
            data.free_sulfur_dioxide,
            data.total_sulfur_dioxide,
            data.density,
            data.pH,
            data.sulphates,
            data.alcohol,
        ]
    ).reshape(1, -1)

    prediction = model.predict(features)

    return int(prediction[0])
