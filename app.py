from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("model.pkl")

# Default homepage
@app.route('/')
def home():
    return "Flask server is running!"

# API route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(input_features).tolist()
    return jsonify({"prediction": prediction})

port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
