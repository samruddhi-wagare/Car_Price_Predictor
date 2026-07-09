from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)   # ✅ Allow frontend to talk to backend

# ===============================
# Load trained model
# ===============================
model = joblib.load("model/car_price_model.pkl")

# ===============================
# Home route (optional but useful)
# ===============================
@app.route("/")
def home():
    return "🚗 Car Price Prediction API is running!"

# ===============================
# Prediction route
# ===============================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Create DataFrame with exact feature names
        input_df = pd.DataFrame([{
            "company": data["company"],
            "year": float(data["year"]),
            "kms_driven": float(data["kms_driven"]),
            "fuel_type": data["fuel_type"]
        }])

        prediction = model.predict(input_df)[0]

        return jsonify({
            "predicted_price": round(float(prediction), 2)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400

# ===============================
# Run server
# ===============================
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
