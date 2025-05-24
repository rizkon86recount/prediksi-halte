import os
import joblib
import numpy as np

def load_model():
    # Mendapatkan path absolut ke file model
    current_dir = os.path.dirname(__file__)  # direktori file predictor.py
    model_path = os.path.join(current_dir, './model_waktu_tempuh.pkl')
    return joblib.load(model_path)

def predict_time(model, data):
    features = np.array([list(data.values())])
    return model.predict(features)[0]
