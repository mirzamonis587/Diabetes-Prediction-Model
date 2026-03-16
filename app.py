from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# lazy load model (render par fast start hota hai)
model = None

def load_model():
    global model
    if model is None:
        model = joblib.load("diabetes_prediction_model.pkl")
    return model


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        Pregnancies = int(request.form["Pregnancies"])
        Glucose = float(request.form["Glucose"])
        BloodPressure = int(request.form["BloodPressure"])
        SkinThickness = float(request.form["SkinThickness"])
        Insulin = float(request.form["Insulin"])
        BMI = float(request.form["BMI"])
        DiabetesPedigreeFunction = float(request.form["DiabetesPedigreeFunction"])
        Age = int(request.form["Age"])

        data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                          Insulin, BMI, DiabetesPedigreeFunction, Age]])

        model = load_model()
        prediction = model.predict(data)

        if prediction[0] == 1:
            result = "Person has Diabetes"
        else:
            result = "Person does not have Diabetes"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")


if __name__ =="__main__":
    app.run(host="0.0.0.0", port=10000)