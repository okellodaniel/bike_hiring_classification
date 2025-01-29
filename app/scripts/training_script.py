import logging
import wget
import pickle
import xgboost as xgb
import pandas as pd
import numpy as np
from os import path
from zipfile import ZipFile
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction import DictVectorizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
MODEL_NAME = '../models/bike_sharing_model.pkl'
FILE_PATH = 'seoul+bike+sharing+demand.zip'
DATA_PATH = f'../data/{FILE_PATH}'
DATA_URL = f'https://archive.ics.uci.edu/static/public/560/{FILE_PATH}'

# Download and extract data
logger.info('Starting data download and extraction')
if not path.exists(DATA_PATH):
    logger.info(f'Downloading dataset from {DATA_URL}')
    wget.download(DATA_URL, DATA_PATH)
    logger.info('Download completed')

with ZipFile(DATA_PATH, 'r') as zip_ref:
    logger.info(f'Extracting {FILE_PATH}')
    zip_ref.extractall()
    logger.info('Extraction completed')

# Load data
logger.info('Loading dataset from CSV')
df = pd.read_csv('./data/SeoulBikeData.csv', encoding='unicode_escape')
logger.info(
    f'Dataset loaded with {len(df)} rows and {len(df.columns)} columns')

# Initial Data Inspection


def inspect_data(df):
    print("First rows inspection:")
    print(df.head())
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nNumerical Statistics:")
    print(df.describe())
    print("\nCategorical Columns:")
    categorical = df.select_dtypes(include='object').columns
    for col in categorical:
        print(f'\nUnique values in {col}:')
        print(df[col].value_counts())


inspect_data(df)

# Data Cleaning


def clean_columns(df):
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df.columns = df.columns.str.replace(r'[()%°/]', '', regex=True)

    categorical = df.select_dtypes(include='object').columns
    for col in categorical:
        df[col] = df[col].str.lower().str.replace(' ', '_')

    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    return df


df = clean_columns(df)

# Feature Engineering


def create_features(df):
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * (df['hour'] / 24))
    df['hour_cos'] = np.cos(2 * np.pi * (df['hour'] / 24))
    return df


df = create_features(df)


# Model Training and Evaluation


def preprocess_data(df):
    df = df.drop(columns=['hour', 'date'])
    return df


df = preprocess_data(df)

# Split data
y = df['rented_bike_count'].values
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(
    df_full_train, test_size=0.25, random_state=42)

y_full_train = df_full_train['rented_bike_count'].values
y_train = df_train['rented_bike_count'].values
y_val = df_val['rented_bike_count'].values
y_test = df_test['rented_bike_count'].values

df_full_train = df_full_train.drop(columns=['rented_bike_count'])
df_train = df_train.drop(columns=['rented_bike_count'])
df_val = df_val.drop(columns=['rented_bike_count'])
df_test = df_test.drop(columns=['rented_bike_count'])

# Model Training and Evaluation Functions


def train_model(df, y, model):
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(df.to_dict(orient='records'))
    model.fit(X_train, y)
    return dv, model


def predict(df, dv, model):
    X = dv.transform(df.to_dict(orient='records'))
    return model.predict(X)


def validate(df_val, y_val, dv, model):
    y_pred = predict(df_val, dv, model)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    return rmse, r2


params = {
    "eta": 0.1,
    "max_depth": 8,
    "min_child_weight": 1,
    "n_estimators": 500,
    "lambda": 0.1,
    "n_jobs": -1
}


def train_final_model(df_train, df_test, y_train, y_test, params):
    model = xgb.XGBRegressor(**params)
    dv, model = train_model(df_train, y_train, model)
    rmse, r2 = validate(df_test, y_test, dv, model)
    return dv, model, rmse, r2


dv, model, rmse, r2 = train_final_model(
    df_full_train, df_test, y_full_train, y_test, params)


def export_model(dv, model, MODEL_NAME):
    logging.info(f'Exporting model to {MODEL_NAME}')
    return pickle.dump((model, dv), open(MODEL_NAME, 'wb'))


export_model(dv, model, MODEL_NAME)

logging.info(f'Final model RMSE: {rmse}')
logging.info(f'Final r2 score: {r2}')
