from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("mlp_model.pkl")
scaler = joblib.load("scaler.pkl")

labels = ["Setosa", "Versicolor", "Virginica"]

@app.get("/")
def home():
    return {"message": "Iris Flower Prediction API"}

@app.get("/predict")
def predict(sepal_length: float,
            sepal_width: float,
            petal_length: float,
            petal_width: float):

    data = np.array([[sepal_length,
                      sepal_width,
                      petal_length,
                      petal_width]])

    data = scaler.transform(data)

    prediction = model.predict(data)[0]

    return {"prediction": labels[prediction]}
