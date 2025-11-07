from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import pandas as pd

pipe = joblib.load("models/pipe.pkl")          # sklearn Pipeline (scaler + model)
column_names = joblib.load("models/column_names.pkl")   # list of feature names

reverse_mapping = {0: "FALSE POSITIVE", 1: "CANDIDATE", 2: "CONFIRMED"}

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        raw_features = [
            request.json["orbital-period"],
            request.json["transit-epoch"],
            request.json["transit-depth"],
            request.json["planet-radius"],
            request.json["semi-major-axis"],
            request.json["inclination"],
            request.json["equilibrium-temp"],
            request.json["insolation-flux"],
            request.json["impact-parameter"],
            request.json["radius-ratio"],
            request.json["stellar-density"],
            request.json["star-distance"],
            request.json["num-transits"],
        ]
        
        df = pd.DataFrame([raw_features], columns=column_names)

        pred = int(pipe.predict(df)[0])
        proba = pipe.predict_proba(df)[0]

        proba_dict = {
            reverse_mapping[i]: round(p, 3) for i, p in enumerate(proba)
        }

        return jsonify(
            {"prediction": reverse_mapping[pred], "probabilities": proba_dict}
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)