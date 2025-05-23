from flask import Blueprint, request, jsonify
from app.model.predictor import load_model, predict_time

bp = Blueprint('main', __name__)
model = load_model()

@bp.route('/predict', methods=['POST'])
def predict():
    data = request.json
    result = predict_time(model, data)
    return jsonify({'predicted_time': result})
