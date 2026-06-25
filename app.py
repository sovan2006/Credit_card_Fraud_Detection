
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("credit_card_fraud_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        transaction_id = int(request.form["transaction_id"])
        amount = float(request.form["amount"])
        transaction_hour = int(request.form["transaction_hour"])
        merchant_category = int(request.form["merchant_category"])
        foreign_transaction = int(request.form["foreign_transaction"])
        location_mismatch = int(request.form["location_mismatch"])
        device_trust_score = float(request.form["device_trust_score"])
        velocity_last_24h = int(request.form["velocity_last_24h"])
        cardholder_age = int(request.form["cardholder_age"])

        input_data = pd.DataFrame({
            "transaction_id": [transaction_id],
            "amount": [amount],
            "transaction_hour": [transaction_hour],
            "merchant_category": [merchant_category],
            "foreign_transaction": [foreign_transaction],
            "location_mismatch": [location_mismatch],
            "device_trust_score": [device_trust_score],
            "velocity_last_24h": [velocity_last_24h],
            "cardholder_age": [cardholder_age]
        })

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(input_data)
        fraud_prob = round(probability[0][1] * 100, 2)

        if prediction == 1:
            result = "🚨 Fraudulent Transaction"
        else:
            result = "✅ Legitimate Transaction"

        return render_template(
            "index.html",
            prediction_text=result,
            fraud_probability=fraud_prob
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True)
