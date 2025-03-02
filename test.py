import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load sample data (Iris dataset)
data = load_iris()
X, y = data.data, data.target

# Train a simple model
model = RandomForestClassifier()
model.fit(X, y)

# Save the model as 'model.pkl'
joblib.dump(model, "model.pkl")

print("Model saved successfully!")
