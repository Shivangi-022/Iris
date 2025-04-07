from flask import Flask, render_template, request
import joblib
import numpy as np
import os
import requests

app = Flask(__name__)

# === Model Download and Load ===

# Google Drive direct download link (not the view link)
model_url = "https://drive.google.com/uc?export=download&id=1oKxvePhj86YR-g6XUgrNoFOuXeL3cI6B"
model_filename = "iris_model_bundle.pkl"

# Download the model only if it doesn't already exist
if not os.path.exists(model_filename):
    print("🔽 Downloading model from Google Drive...")
    response = requests.get(model_url)
    with open(model_filename, "wb") as f:
        f.write(response.content)
    print("✅ Model downloaded.")

# Load model components
print("📦 Loading model...")
bundle = joblib.load(model_filename)
model = bundle['model']
scaler = bundle['scaler']
label_encoder = bundle['label_encoder']
print("✅ Model ready.")

# === Routes ===

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Convert form values to float list
        features = [float(x) for x in request.form.values()]
        scaled_features = scaler.transform([features])
        pred_class = model.predict(scaled_features)
        pred_species = label_encoder.inverse_transform(pred_class)[0]

        return render_template("index.html", prediction_text=f"🌸 Predicted species: {pred_species}")
    except Exception as e:
        return render_template("index.html", prediction_text=f"❌ Error: {str(e)}")

# === Main ===

if __name__ == "__main__":
    app.run()
