import pickle
import os
import logging

from flask import Flask, request, jsonify, render_template
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

app = Flask(__name__)
logger.info(f'Starting app')

# Constants
MODEL_FILE = os.path.join(os.path.dirname(
    __file__), 'models/bike_sharing_model.pkl')
logger.info(f'Loading model from {MODEL_FILE}')

# app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:postgres@db:5432/bikeshare'
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:postgres123@localhost:5432/bikeshare'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# logger.info(db.ping(reconnect=True))
logger.info('Database connected')


@app.route('/', methods=['POST'])
def index():
    if request.method == 'POST':
        ride = request.get_json()
        logger.info('Ride data: %s', ride)

        predict = predict_rented_bike_count(ride)[0].round(0)
        logger.info('Predicted count: %s', predict)

        response = {
            'predicted_count': float(predict)
        }

        return jsonify(response)


@app.route('/', methods=['GET'])
def dashboard():
    result = BikeShare.query.all()
    return render_template('dashboard.html', data=result)


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


class BikeShare(db.Model):
    __tablename__ = 'bikeshares'
    id = db.Column(db.Integer, primary_key=True)
    temperature = db.Column(db.Float, nullable=False)
    windspeed = db.Column(db.Float, nullable=False)
    visibility = db.Column(db.Integer, nullable=False)
    dewpointtemperature = db.Column(db.Float, nullable=False)
    solarradiation = db.Column(db.Float, nullable=False)
    rainfall = db.Column(db.Float, nullable=False)
    snowfall = db.Column(db.Float, nullable=False)
    seasons = db.Column(db.Float, nullable=False)
    holiday = db.Column(db.String(20), nullable=False)
    functioningday = db.Column(db.String(4), nullable=False)
    month = db.Column(db.Integer, nullable=False)
    day = db.Column(db.Integer, nullable=False)
    dayofweek = db.Column(db.Integer, nullable=False)
    hoursin = db.Column(db.Float, nullable=False)
    hourcos = db.Column(db.Float, nullable=False)

    def __repr__(self):
        return f'<BikeShare {self.id}>'


def create_database():
    with app.app_context():
        db.create_all()
        print("Database and tables created successfully!")


if __name__ == "__main__":
    create_database()
    app.run()
