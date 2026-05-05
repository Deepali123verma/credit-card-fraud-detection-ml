from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("models/model.pkl")

@app.get("/")
def home():
    return {"message": "Fraud Detection API Running"}

@app.post("/predict")
def predict(data: list):
    data = np.array(data)
    pred = model.predict(data)
    return {"prediction": pred.tolist()}