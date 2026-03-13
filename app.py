import numpy as np
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI
import joblib

# Create FastAPI app
app = FastAPI()

# Load trained model
model = joblib.load("diabetes_prediction_model.pkl")

# Input data schema
class InputData(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int


# Home route
@app.get("/")
def home():
    return {"message": "ML model API running successfully"}


# Prediction route
@app.post("/predict")
def predict(data: InputData):

    input_data = [[
        data.Pregnancies,
        data.Glucose,
        data.BloodPressure,
        data.SkinThickness,
        data.Insulin,
        data.BMI,
        data.DiabetesPedigreeFunction,
        data.Age
    ]]

    prediction = model.predict(input_data)

    return {"Prediction": int(prediction[0])}