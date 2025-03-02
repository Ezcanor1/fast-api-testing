from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if "features" not in data:
            return jsonify({"error": "Missing 'features' key in request"}), 400

        input_features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(input_features).tolist()
        
        return jsonify({"prediction": prediction})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Returns error message

port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
