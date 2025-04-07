from flask import Flask, render_template, request
import joblib
import numpy as np

# Load model bundle
bundle = joblib.load("iris_model_bundle.pkl")
model = bundle['model']
scaler = bundle['scaler']
label_encoder = bundle['label_encoder']

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        scaled_features = scaler.transform([features])
        pred_class = model.predict(scaled_features)
        pred_species = label_encoder.inverse_transform(pred_class)[0]
        return render_template("index.html", prediction_text=f"Predicted species: {pred_species}")
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run()
