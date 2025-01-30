import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

# Initialize Flask app
app = Flask(__name__)

# Configure the database connection
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv(
    'DATABASE_URI', 'postgresql://postgres:postgres123@localhost:5432/bike_sharing'
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Define the RideData model (same as in your app.py)


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

# Create the database and tables


def create_database():
    with app.app_context():
        db.create_all()
        print("Database and tables created successfully!")


if __name__ == "__main__":
    create_database()
