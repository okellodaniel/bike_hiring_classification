import pickle
import os
import logging


from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy


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
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://user:password@db:5432/bike_sharing'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


class RideData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    temperature_c = db.Column(db.Float)
    humidity = db.Column(db.Integer)
    wind_speed_ms = db.Column(db.Float)
    visibility_10m = db.Column(db.Integer)
    dew_point_temperature_c = db.Column(db.Float)
    solar_radiation_mjm2 = db.Column(db.Float)
    rainfallmm = db.Column(db.Float)
    snowfall_cm = db.Column(db.Float)
    seasons = db.Column(db.String(50))
    holiday = db.Column(db.String(50))
    functioning_day = db.Column(db.String(50))
    month = db.Column(db.Integer)
    day = db.Column(db.Integer)
    dayofweek = db.Column(db.Integer)
    hour_sin = db.Column(db.Float)
    hour_cos = db.Column(db.Float)
    prediction = db.Column(db.Float)


logger.info(f'Starting app')


@app.route('/', methods=['POST'])
def index():
    if request.method == 'POST':
        ride = request.get_json()
        logger.info('Ride data: %s', ride)

        predict = predict_rented_bike_count(ride)
        logger.info('Predicted count: %s', predict)

        # save ride data to database together with prediction

        new_ride = RideData(**ride)
        new_ride.prediction = predict

        db.session.add(new_ride)
        logger.info(f'Saving ride data to database')
        db.session.commit()

        return jsonify({'predicted_count': predict})


def predict_rented_bike_count(ride):
    model, dv = load_model(MODEL_FILE)
    X_ride = dv.transform(ride)
    y_pred = model.predict(X_ride)
    return y_pred


def load_model(file_name):
    with open(file_name, 'rb') as f:
        model,dv = pickle.load(f)
    logger.info("Loading model and data vectorizer")
    return model, dv


if __name__ == "__main__":
    app.run(debug=True,
            host='0.0.0.0',
            port=9696)
