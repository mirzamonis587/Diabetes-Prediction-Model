from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import joblib

app = Flask(__name__)
app.secret_key = "secret123"

# load model
model = None

def load_model():
    global model
    if model is None:
        model = joblib.load("diabetes_prediction_model.pkl")
    return model


@app.route("/")
def home():
    prediction = session.pop("prediction", None)
    return render_template("index.html", prediction_text=prediction)


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

        prediction = model.predict(data)

        if prediction[0] == 1:
            result = "Person has Diabetes"
        else:
            result = "Person does not have Diabetes"

        session["prediction"] = result

    except Exception as e:
        session["prediction"] = f"Error: {str(e)}"

    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run()