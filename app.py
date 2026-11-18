
import os
import joblib
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

# Load trained model and preprocessing pipeline
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "fraud_detection_model.joblib"
)

saved_model = joblib.load(MODEL_PATH)

preprocessor = saved_model["preprocessor"]
model = saved_model["model"]
threshold = saved_model["threshold"]

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        transaction = {
            "step": int(request.form["step"]),
            "customer": request.form["customer"],
            "age": request.form["age"],
            "gender": request.form["gender"],
            "merchant": request.form["merchant"],
            "category": request.form["category"],
            "amount": float(request.form["amount"])
        }

        input_df = pd.DataFrame([transaction])

        processed_data = preprocessor.transform(input_df)

        probability = model.predict_proba(processed_data)[0][1]

        prediction = (
            "Fraudulent Transaction"
            if probability >= threshold
            else "Legitimate Transaction"
        )

        return render_template(
            "index.html",
            prediction=prediction,
            probability=round(probability * 100, 2)
        )

    except Exception as e:
        return render_template(
            "index.html",
            error=str(e)
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
