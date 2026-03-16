from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# load trained model
model = joblib.load("diabetes_prediction_model.pkl")


@app.route("/")
def home():
    return render_template("index.html",prediction_text =None)


@app.route("/predict", methods=["POST"])
def predict():

    Pregnancies = int(request.form["Pregnancies"])
    Glucose = float(request.form["Glucose"])
    BloodPressure = int(request.form["BloodPressure"])
    SkinThickness = float(request.form["SkinThickness"])
    Insulin = float(request.form["Insulin"])
    BMI = float(request.form["BMI"])
    DiabetesPedigreeFunction = float(request.form["DiabetesPedigreeFunction"])
    Age = int(request.form["Age"])

    # convert input into numpy array
    data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                      Insulin, BMI, DiabetesPedigreeFunction, Age]])

    prediction = model.predict(data)

    if prediction[0] == 1:
        result = "Person has Diabetes"
    elif prediction[0] == 0:
        result = "Person does not have Diabetes"
    else:
        return render_template("index.html", prediction_text=result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
   


    

