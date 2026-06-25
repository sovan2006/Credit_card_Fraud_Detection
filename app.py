from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model only
model = joblib.load("credit_card_fraud_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = pd.DataFrame({
            "transaction_id": [int(request.form["transaction_id"])],
            "amount": [float(request.form["amount"])],
            "transaction_hour": [int(request.form["transaction_hour"])],
            "merchant_category": [int(request.form["merchant_category"])],
            "foreign_transaction": [int(request.form["foreign_transaction"])],
            "location_mismatch": [int(request.form["location_mismatch"])],
            "device_trust_score": [float(request.form["device_trust_score"])],
            "velocity_last_24h": [int(request.form["velocity_last_24h"])],
            "cardholder_age": [int(request.form["cardholder_age"])]
        })

        prediction = model.predict(input_data)[0]

        if hasattr(model, "predict_proba"):
            fraud_prob = round(
                model.predict_proba(input_data)[0][1] * 100,
                2
            )
        else:
            fraud_prob = None

        result = (
            "🚨 Fraudulent Transaction"
            if prediction == 1
            else "✅ Legitimate Transaction"
        )

        return render_template(
            "index.html",
            prediction_text=result,
            fraud_probability=fraud_prob
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {e}"
        )


if __name__ == "__main__":
    app.run(debug=True)