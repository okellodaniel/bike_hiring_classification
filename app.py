import pickle
import os
import logging


from flask import Flask, request, jsonify

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Constants
MODEL_FILE = os.path.join(os.path.dirname(
    __file__), 'models/bike_sharing_model.pkl')
logger.info(f'Loading model from {MODEL_FILE}')

app = Flask(__name__)


logger.info(f'Starting app')


@app.route('/', methods=['POST'])
def index():
    if request.method == 'POST':
        ride = request.get_json()
        logger.info('Ride data: %s', ride)

        predict = predict_rented_bike_count(ride)
        logger.info('Predicted count: %s', predict)

        return jsonify({'predicted_count': predict})


def predict_rented_bike_count(ride):
    model, dv = load_model(MODEL_FILE)
    X_ride = dv.transform(ride)
    y_pred = model.predict(X_ride)
    return y_pred


def load_model(file_name):
    with open(file_name, 'rb') as f:
        model, dv = pickle.load(f)
    logger.info("Loading model and data vectorizer")
    return model, dv


if __name__ == "__main__":
    app.run(debug=True,
            host='0.0.0.0',
            port=9696)
