import pickle
import logging

import xgboost as xgb
from flask import Flask
from sklearn import

app = Flask(__name__)


@app.route('/')
def predict_rented_bike_count():
    return 'Hello, World!'
