import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import model_from_json

# ---------------------------
# LOAD DATASET (for scaler)
# ---------------------------
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)

X = df.drop("Outcome", axis=1)

# ---------------------------
# SCALE DATA
# ---------------------------
scaler = StandardScaler()
scaler.fit(X)

# ---------------------------
# LOAD & CLEAN MODEL JSON
# ---------------------------
with open("model.json", "r") as f:
    model_dict = json.load(f)

def remove_quantization(obj):
    if isinstance(obj, dict):
        obj.pop("quantization_config", None)
        for key in obj:
            remove_quantization(obj[key])
    elif isinstance(obj, list):
        for item in obj:
            remove_quantization(item)

remove_quantization(model_dict)

model_json = json.dumps(model_dict)

# ---------------------------
# LOAD MODEL
# ---------------------------
model = model_from_json(model_json)
model.load_weights("model.weights.h5")

# ---------------------------
# PREDICTION FUNCTION
# ---------------------------
def predict_diabetes(input_data):
    input_data = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_data)
    input_scaled = input_scaled.reshape(-1, 8, 1)

    prediction = model.predict(input_scaled)

    prob = prediction[0][0]

    # Return BOTH result + probability
    if prob > 0.5:
        return "High Risk", prob
    else:
        return "Low Risk", prob