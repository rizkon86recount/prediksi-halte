import joblib
import numpy as np

def load_model():
    return joblib.load('app/model/model_waktu_tempuh.pkl')

def predict_time(model, data):
    features = np.array([list(data.values())])
    return model.predict(features)[0]
