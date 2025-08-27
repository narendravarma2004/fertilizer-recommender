import os
from flask import Flask, render_template, request
import tensorflow as tf
import joblib
import numpy as np
import json

# -------------------------------
# Suppress TensorFlow warnings
# -------------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=INFO, 2=WARNING, 3=ERROR

# -------------------------------
# Load trained model and encoders
# -------------------------------
model = tf.keras.models.load_model("fertilizer_model.h5")
label_encoders = joblib.load("label_encoders.pkl")
fertilizer_encoder = joblib.load("fertilizer_encoder.pkl")

# -------------------------------
# Load dropdown options
# -------------------------------
with open("options.json", "r") as f:
    options = json.load(f)

# -------------------------------
# Initialize Flask app
# -------------------------------
app = Flask(__name__)

# -------------------------------
# Routes
# -------------------------------
@app.route("/")
def home():
    return render_template(
        "index.html",
        soil_types=options["soil_types"],
        crop_types=options["crop_types"]
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values from form
        temperature = float(request.form["temperature"])
        moisture = float(request.form["moisture"])
        soil_type = request.form["soil_type"]
        crop_type = request.form["crop_type"]
        nitrogen = float(request.form["nitrogen"])
        potassium = float(request.form["potassium"])
        phosphorous = float(request.form["phosphorous"])

        # Encode categorical values
        soil_encoded = label_encoders["Soil Type"].transform([soil_type])[0]
        crop_encoded = label_encoders["Crop Type"].transform([crop_type])[0]

        # Prepare input for model
        input_data = np.array([[temperature, moisture, soil_encoded, crop_encoded,
                                nitrogen, potassium, phosphorous]])

        # Make prediction
        prediction = model.predict(input_data)
        predicted_class = np.argmax(prediction, axis=1)[0]
        fertilizer_name = fertilizer_encoder.inverse_transform([predicted_class])[0]

        return render_template(
            "index.html",
            soil_types=options["soil_types"],
            crop_types=options["crop_types"],
            result=f"🌱 Recommended Fertilizer: {fertilizer_name}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            soil_types=options["soil_types"],
            crop_types=options["crop_types"],
            result=f"❌ Error: {str(e)}"
        )

# -------------------------------
# Run locally only
# Render will use Gunicorn
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
