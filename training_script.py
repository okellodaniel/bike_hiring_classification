import logging
import wget
import json
import pickle
import xgboost as xgb
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from os import path
from zipfile import ZipFile
from sklearn.feature_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_extraction import DictVectorizer
from itertools import product

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
FILE_PATH = 'seoul+bike+sharing+demand.zip'
DATA_URL = f'https://archive.ics.uci.edu/static/public/560/{FILE_PATH}'

# Download and extract data
logger.info('Starting data download and extraction')
if not path.exists(FILE_PATH):
    logger.info(f'Downloading dataset from {DATA_URL}')
    wget.download(DATA_URL)
    logger.info('Download completed')

with ZipFile(FILE_PATH, 'r') as zip_ref:
    logger.info(f'Extracting {FILE_PATH}')
    zip_ref.extractall()
    logger.info('Extraction completed')

# Load data
logger.info('Loading dataset from CSV')
df = pd.read_csv('./SeoulBikeData.csv', encoding='unicode_escape')
logger.info(
    f'Dataset loaded with {len(df)} rows and {len(df.columns)} columns')

# Initial Data Inspection


def inspect_data(df):
    print("First rows inspection:")
    display(df.head())
    print("\nData Types:")
    display(df.dtypes)
    print("\nMissing values:")
    display(df.isnull().sum())
    print("\nNumerical Statistics:")
    display(df.describe())
    print("\nCategorical Columns:")
    categorical = df.select_dtypes(include='object').columns
    for col in categorical:
        print(f'\nUnique values in {col}:')
        display(df[col].value_counts())


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

# Exploratory Data Analysis


def plot_eda(df):
    # Rented Bike count per season
    plt.figure(figsize=(16, 10))
    sns.boxplot(data=df, x='seasons', y='rented_bike_count')
    plt.xticks(rotation=45)
    plt.title('Bike Rentals by Season')
    plt.show()

    # Average rentals per hour
    hourly_avg = df.groupby('hour')['rented_bike_count'].mean()
    plt.figure(figsize=(16, 10))
    hourly_avg.plot()
    plt.title('Average Rentals by Hour')
    plt.xlabel('Hour')
    plt.ylabel('Average Rentals')
    plt.show()

    # Temperature vs Rentals
    plt.figure(figsize=(12, 5))
    sns.scatterplot(data=df, x='temperature_c',
                    y='rented_bike_count', alpha=0.5)
    plt.title('Temperature Impact on Bike Rentals')
    plt.show()

    # Weather Impact Analysis
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    sns.scatterplot(data=df, x='temperature_c',
                    y='rented_bike_count', ax=axes[0, 0])
    axes[0, 0].set_title('Temperature vs Rentals')
    sns.scatterplot(data=df, x='humidity',
                    y='rented_bike_count', ax=axes[0, 1])
    axes[0, 1].set_title('Humidity vs Rentals')
    sns.scatterplot(data=df, x='wind_speed_ms',
                    y='rented_bike_count', ax=axes[1, 0])
    axes[1, 0].set_title('Wind Speed vs Rentals')
    sns.scatterplot(data=df, x='solar_radiation_mjm2',
                    y='rented_bike_count', ax=axes[1, 1])
    axes[1, 1].set_title('Solar Radiation vs Rentals')
    plt.show()

    # Holiday Impact Analysis
    plt.figure(figsize=(6, 5))
    sns.boxplot(data=df, x='holiday', y='rented_bike_count')
    plt.title('Rental Patterns: Holiday vs Non-Holiday')
    plt.show()

    # Correlation Analysis
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    df_numerical = df[numerical_cols].drop(columns=['hour_cos', 'hour_sin'])
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_numerical.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix Numerical Features')
    plt.show()


plot_eda(df)

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


# Models and Parameters
models = {
    'xgboost': {
        'model': xgb.XGBRegressor,
        'params': {
            'max_depth': [3, 5, 6, 7, 8, 9],
            'n_estimators': [50, 100, 200, 300, 500],
            'eta': [0.01, 0.1, 0.3, 0.5],
            'min_child_weight': [1, 2, 3, 4, 5],
            'lambda': [0.1, 0.5, 0, 1, 2]
        }
    },
    'random_forest': {
        'model': RandomForestRegressor,
        'params': {
            'max_depth': [10, 20, 30, 50],
            'n_estimators': [50, 100, 200, 300, 500],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4, 7]
        }
    },
    'decision_tree': {
        'model': DecisionTreeRegressor,
        'params': {
            'max_depth': [5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 3, 4, 5],
            'max_features': ['sqrt', 'log2']
        }
    },
    'linear': {
        'model': LinearRegression,
        'params': {
            'fit_intercept': [True, False]
        }
    }
}


def tune_models(df_train, y_train, df_val, y_val, models):
    results = {}
    for model_name, model_info in models.items():
        logger.info(f'Starting hyperparameter tuning for {model_name}')
        best_rmse = float('inf')
        best_params = None
        param_combinations = [dict(zip(model_info['params'].keys(), v))
                              for v in product(*model_info['params'].values())]

        logger.info(
            f'Evaluating {len(param_combinations)} parameter combinations')

        for i, params in enumerate(param_combinations, 1):
            model = model_info['model'](**params)
            dv, model = train_model(df_train, y_train, model)
            y_pred = predict(df_val, dv, model)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))

            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params
                logger.debug(f'New best RMSE {rmse:.4f} with params {params}')

            if i % 10 == 0:
                logger.info(
                    f'Completed {i}/{len(param_combinations)} combinations')

        logger.info(
            f'Best {model_name} RMSE: {best_rmse:.4f} with params {best_params}')
        results[model_name] = {
            'best_params': best_params,
            'best_rmse': best_rmse
        }
    return results


def train_with_tuning(df_full_train, df_train, df_val, df_test, y_full_train, y_train, y_val, y_test):
    tuning_results = tune_models(df_train, y_train, df_val, y_val, models)
    final_models = {}

    for model_name, result in tuning_results.items():
        print(f'Training final model: {model_name} with best parameters')
        model = models[model_name]['model'](**result['best_params'])
        dv, model = train_model(df_full_train, y_full_train, model)
        test_rmse, test_r2 = validate(df_test, y_test, dv, model)

        final_models[model_name] = {
            'model': model,
            'dv': dv,
            'test_rmse': test_rmse,
            'test_r2': test_r2
        }
    return final_models, tuning_results


final_models, tuning_results = train_with_tuning(
    df_full_train, df_train, df_val, df_test, y_full_train, y_train, y_val, y_test)

# Save the best model
best_model = final_models['xgboost']['model']
best_dv = final_models['xgboost']['dv']
pickle.dump((best_model, best_dv), open('xgboost_0_94.pkl', 'wb'))

# Test the model
sample = df_test.sample(1).to_dict(orient='records')
X_sample = best_dv.transform(sample)
y_pred_sample = best_model.predict(X_sample)
print(f'Predicted rental count: {y_pred_sample[0]}')
