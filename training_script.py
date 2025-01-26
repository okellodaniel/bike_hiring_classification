import wget
import json
import pickle
import xgboost as xgb
from os import path
from zipfile import ZipFile as zipfile
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


file_path = 'seoul+bike+sharing+demand.zip'
if not path.exists(file_path):
    wget.download(f'https://archive.ics.uci.edu/static/public/560/{file_path}')


with  zipfile(file_path, 'r') as zip_ref:
    zip_ref.extractall()


df = pd.read_csv('./SeoulBikeData.csv',encoding='unicode_escape')


# First rows inspection
df.head()

# Data Types
df.dtypes

# Missing values
df.isnull().sum()

# Numerical Statistics
df.describe()

# Inspect categorical columns
df_copy = df.copy()
categorical = df_copy.dtypes[df_copy.dtypes == 'object'].index
df_categorical = df_copy[categorical].drop(columns='Date')

for col in df_categorical:
    print(f'\nUnique values in col: {col}')
    display(df_categorical[col].value_counts())


"""
Data Cleaning
"""

df.columns = df.columns.str.lower().str.replace(' ','_')


cols = df.columns
cols


def clean_columns(df):
    cleaned_cols = []
    
    for col in df.columns:
        clean_col = col.lower()
        clean_col = clean_col.replace('(', '').replace(')', '').replace('%','').replace('/','').replace('°',' ')
        clean_col = clean_col.replace(' ','_')
        cleaned_cols.append(clean_col)
        
    df.columns = cleaned_cols

    # clean categorical column data
    categorical = df.dtypes[df.dtypes == 'object'].index

    for col in categorical:
        df[col] = df[col].str.lower().str.replace(' ','_')

    df.date = pd.to_datetime(df.date, format='%d/%m/%Y')
    return df
clean_columns(df)

df.functioning_day.value_counts()


"""
Data Preprocessing
  Create temporal Features
 - Month
 - Day
 - Day Of Week
  Create Cyclic features for hour to represent hourly cyclic nature
 - Hour sin representation
 - Hour cos representation
"""

# Temporal features
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek


# cyclic features
df['hour_sin'] = np.sin(2*np.pi*(df.hour/24))
df['hour_cos'] = np.cos(2*np.pi*(df.hour/24))


"""
Exploratory Data Analysis
"""

numerical_cols = list(df.dtypes[(df.dtypes == 'int64') | (df.dtypes == 'float64')].index)
numerical_cols


# Rented Bike count per season
plt.figure(figsize=(16,10))
plt.subplot(2,2,1)
sns.boxplot(data=df,x='seasons',y='rented_bike_count')
plt.xticks(rotation=45)
plt.title('Bike Rentals by season')

"""
**Observations**
 - Summer has the highest median rentals (900) and widest spread.
 - Spring and Autumn show similar patterns with moderate rentals, 600 and 700 respectively.
 - Winter has significantly lower rentals (250 bikes) and the smallest spread
 
 **Key Insights**
 - Seasonality has a strong impact on bike rentals
 - Summer shows the highest variability in rental numbers
 - Winter has more consistent rental patterns
 - All seasons show outliers at the on the higher end, indicating occassional peaks in demand
 
 """

# Average rentals per hour
hourly_avg = df.groupby('hour')['rented_bike_count'].mean()
plt.figure(figsize=(16,10))
plt.subplot(2,2,3)
hourly_avg.plot()
plt.title('Average rentals by hour')
plt.xlabel('Hour')
plt.ylabel('Average rentals')


"""
Key Insights
1. Peak Hours
    - Morning peak - Around 8:00am with about 1000 rental bikes
    - Evening Peak - Around 6:00pm with about 1500 bikes
    - The evening peak is higher than the morning peak
    - These peaks clearly align with commuting hours
2. Low Usage Periods
    - Lowest usage - From 4-5 am with around 150 rentals
    - A gradual decrese from midnight to early morning, representing a quiet period.
3. Daily Pattern
    - M shape pattern showing 2 major peaks
    - Mid-day plateau between 10-15 hours showing steady moderate usage 
"""

# Temperature vs Rentals

plt.figure(figsize=(12,5))
sns.scatterplot(data=df, x='temperature_c',y='rented_bike_count',alpha=0.5)
plt.title('Temperature Impact on Bike Rentals')


"""
Key Insights
- Clear positive correlation between temperature and rentals
- Non linear relationship, where rentals increase more rapidly between 0-20 degrees 
- Highest rental activities occur around 20 to 30 degrees, suggesting other factors affect rentals
"""


"""
Weather Impact Analysis
"""

# Weather conditions combined analysis
plt.figure(figsize=(15,5))
fig,axes = plt.subplots(2,2,figsize=(12,10))

# Temparature
sns.scatterplot(data=df,x='temperature_c',y='rented_bike_count',ax=axes[0,0])
axes[0,0].set_title('Temperature vs Rentals')

# Humidity
sns.scatterplot(data=df,x='humidity',y='rented_bike_count',ax=axes[0,1])
axes[0,1].set_title('Humidity vs Rentals')

# Wind Speed
sns.scatterplot(data=df,x='wind_speed_ms',y='rented_bike_count',ax=axes[1,0])
axes[1,0].set_title('WindSpeed vs Rentals')

# Wind Speed
sns.scatterplot(data=df,x='solar_radiation_mjm2',y='rented_bike_count',ax=axes[1,1])
axes[1,1].set_title('Solar Radiation (MJ/m2) vs Rentals')

"""
Holiday Impact Analysis
"""

# Holiday vs Non-holiday comparison

plt.figure(figsize=(6,5))
sns.boxplot(data=df,x='holiday',y='rented_bike_count')
plt.title('Rental patterns: Holiday vs Non-Holiday')


"""
 Observations
 - Non holiday rentals show a higher median ~500 rentals, with a larger interquatile range. There are sights of higher values (up to 3500) with more outliers at higher values.
 - Holiday rentals show lower median rentals ~250 bikes, with a smaller interquartile range, with fewer outliers

"""
# Hourly rental patterns: Holiday vs Non holiday

plt.figure(figsize=(12,6))
holiday_hourly = df.pivot_table(
    index='hour',
    columns='holiday',
    values='rented_bike_count',
    aggfunc='mean'
)
holiday_hourly.plot(
    figsize=(12,6)
)
plt.title('Hourly Rental Patterns: Holiday vs Non-Holiday')


df_num = df.copy()

numerical_cols = df_num.select_dtypes(include=[np.number]).columns
df_numerical = df_num[numerical_cols].drop(columns=['hour_cos','hour_sin'])

"""
 Insights
 
 - Non-holidays show two distinct peaks (commuting hours), morning peak between 8 and 9 am ~1000 rentals, Evening peak around 6pm ~1500
 - Holidays, show no clear commuting peaks, more less a gradual increase through the day. Peak around 5 to 6pm but much lower than non holidays
"""

"""
 Correlation Analysis
"""


plt.figure(figsize=(12,8))
sns.heatmap(df_numerical.corr(),annot=True,cmap='coolwarm',center=0)
plt.title('Correlation Matrix Numerical Features')


# Strongest correlation with Bike Rentals
# - Temperature - 0.54, Strongest positive correlation - People rent more bikes in warmer weather
# - Hour - 0.42, Strong positive correlation - Indicating clear daily plans.
# - Dew Point - 0.38, moderate positive correlation
# - Solar radioation - 0.26, Weak positive correlation - more rentals during sunny conditions
# - Humidity - -0.2, Weak negative correlation - fewer rentals when humidity is high.
# 
# Further Insights
# - Temperature and Dew point show a very strong correlation (0.91)
#     - This suggests multicollinearity which might neccessitate dropping one of them. Since Temperature has a stronger correlation with the rentals, I opt to keep it.
# - Humidity and visibility show a strong negative correlation (-0.54), meaning the higher the humidity the the lower the visibility.


"""
Distribution analysis
"""

# Distribution of target variable
plt.figure(figsize=(12,6))
sns.histplot(data=df,x='rented_bike_count',bins=50,kde=True)
plt.title('Distribution of Bike Rentals')


# Observations
# - Right skewed with many low rental counts, with a long tail towards higher rental counts.
# - This is not normally distributed


# Log Transformation to check if more normally distributed
plt.figure(figsize=(12,6))
sns.histplot(data=df,x=np.log1p(df.rented_bike_count),bins=50,kde=True)
plt.title('Distribution of Log-Transformed Bike Rentals')


# A more symetric but still not perfectly normal, with better speared of values

# ## Training Model


get_ipython().system('pip install xgboost')


from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from itertools import product


# #### Splitting dataset

df.columns


def preprocess_data(df):
    df = df.copy()
    df_processed = df.drop(columns=['hour','date'])
    return df_processed
df = preprocess_data(df)


y = df.rented_bike_count.values

df_full_train,df_test = train_test_split(
    df,test_size=0.2,random_state=42
)

df_train,df_val = train_test_split(
    df_full_train,test_size=0.25,random_state=42
)



y_full_train = df_full_train.rented_bike_count.values
y_train = df_train.rented_bike_count.values
y_val = df_val.rented_bike_count.values
y_test = df_test.rented_bike_count.values


del df_full_train['rented_bike_count']
del df_train['rented_bike_count']
del df_val['rented_bike_count']
del df_test['rented_bike_count']


# #### Training function

def train_model(df,y,model):
    dict = df.to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dict)
    
    model.fit(X_train,y)

    return dv,model


# #### Prediction Function

def predict(df,dv,model):
    val_dicts = df.to_dict(orient='records')
    X = dv.transform(val_dicts)
    y_pred = model.predict(X)
    return y_pred


# #### Validation Function

# In[59]:


def validate(df_val,y_val,dv,model):
    y_pred = predict(df_val,dv,model)
    rmse = np.sqrt(mean_squared_error(y_val,y_pred))
    r2 = r2_score(y_val,y_pred)
    return rmse,r2


# ### Models
# 
# My approach involves training 
# - xgboost regressor
# - random forest regressor
# - linear regression
# - decision tree regressor
# 
# However the approach involves training all models using GridSearchCV, factoring parameter tuning

# In[61]:


# Models and params
models = {
        'xgboost':{
            'model':xgb.XGBRegressor,
            'params':{
                'max_depth': [3,5,6,7,8,9],
                'n_estimators': [50,100,200,300,500],
                'eta': [0.01,0.1,0.3,0.5],
                'min_child_weight': [1,2,3,4,5],
                'lambda':[0.1,0.5,0,1,2]
            }
        },
        'random_forest':{
            'model': RandomForestRegressor,
            'params':{
                'max_depth':[10,20,30,50],
                'n_estimators':[50,100,200,300,500],
                'min_samples_split':[2,5,10],
                'min_samples_leaf':[1,2,4,7]
                
            }
        },
        'decision_tree':{
            'model': DecisionTreeRegressor,
            'params':{
                'max_depth': [5,10,15,20],
                'min_samples_split': [2,5,10],
                'min_samples_leaf': [1,2,3,4,5],
                'max_features':['sqrt','log2']
            }
        },
        'linear':{
            'model': LinearRegression,
            'params':{
                'fit_intercept':[True, False]
            }
        }
    }


def tune_models(df_train,y_train,df_val,y_val,models):
    dv = DictVectorizer(sparse=False)
    train_dicts = df_train.to_dict(orient='records')
    val_dicts = df_val.to_dict(orient='records')

    X_train = dv.fit_transform(train_dicts)
    X_val = dv.fit_transform(val_dicts)

    results = {}

    for model_name, model_info in models.items():
        print(f'Tuning model :{model_name}')

        best_rmse = float('inf')
        best_params = None

        # Generate all parameter combos
        param_combinations = [dict(zip(model_info['params'].keys(),v)) for v in product(*model_info['params'].values())]

        for params in param_combinations:
            model = model_info['model'](**params)
            # print(f'\nmodel: {model}')
            # print(f'\nmodel parameters: {params}')

            if model_name == 'xgboost':
                print('xgboost')
                model.fit(
                    X_train,y_train,
                    eval_set=[(X_val,y_val)],
                    verbose=False
                )
            else:
                model.fit(X_train,y_train)
            
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val,y_pred))

            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params

            print(f'parameters: {params} -> RMSE: {rmse:.3f}')
        results[model_name] = {
            'best_params': best_params,
            'best_rmse': best_rmse,
            'parameter_hidtory': param_combinations,
            'rmse_history': rmse
        }    

    return results,dv


#Train Models with parameter tuning
def train_with_tuning(df_full_train,df_train,df_val,df_test,y_full_train,y_train,y_val,y_test):
    tuning_results,dv = tune_models(
        df_train,
        y_train,
        df_val,
        y_val,
        models
    )
    
    final_models = {}
    for model_name,result in tuning_results.items():
        print(f'Training final model: {model_name} model with best parameters')
        # print(f"Best parameters: {result['best_params']}")

        if model_name == 'xgboost':
            model = xgb.XGBRegressor(**result['best_params'])
        elif model_name == 'random_forest':
            model = RandomForestRegressor(**result['best_params'])
        elif model_name == 'decision_tree':
            model = DecisionTreeRegressor(**result['best_params'])
        else:
            model = LinearRegression(**result['best_params'])
        final_dv, final_model = train_model(df_full_train,y_full_train,model)
        test_rmse,test_r2 = validate(df_test,y_test,final_dv,final_model)

        print(f"Test RMSE: {test_rmse:.3f}")
        print(f"Test r2: {test_r2:.4f}")

        final_models[model_name] = {
            'model': final_model,
            'dv': final_dv,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
        }
    return final_models,tuning_results


final_models,tuning_results = train_with_tuning(df_full_train,df_train,df_val,df_test,y_full_train,y_train,y_val,y_test)
final_models,tuning_results





print(final_models)


# ### Insights
# 
# 1. **Model Comparison**
#    - Four models were trained: XGBoost, Random Forest, Decision Tree, and Linear Regression
#    - Each model has a DictVectorizer for feature transformation
# 
# 2. **Performance Metrics**
#    - RMSE (Root Mean Square Error) - lower is better
#    - R² (R-squared) - higher is better, max is 1.0
# 
# 3. **Rankings (best to worst)**:
# 
# 
# |  Model   | RMSE   | R²   |
# |  -------  |------  |-----  |
# |   XGBoost  | 148.23  | 0.947   |
# |   Random Forest   | 178.35   | 0.924   |
# |   Decision Tree   | 351.77   | 0.703   |
# |   Linear   | 445.93   | 0.523   |
# 
# 4. **XGBoost Configuration**:
#    - 500 estimators
#    - Max depth: 8
#    - Learning rate (eta): 0.1
#    - Lambda (L2): 0.1
# 
# XGBoost is clearly the best performing model with 94.7% of variance as in the final_models. The linear model performed poorly, suggesting non-linear relationships in the data.

# print(tuning_results)
print(json.dumps(tuning_results))


# ## Final Model


model = final_models['xgboost']['model']
dv = final_models['xgboost']['dv']


X_full_encoded = dv.transform(df_full_train.to_dict(orient='records'))
X_test_encoded = dv.fit_transform(df_test.to_dict(orient='records'))


model = model.fit(X_full_encoded,y_full_train)
model


model_name = 'xgboost_0_94.pkl'
pickle.dump((model,dv),open(model_name,'wb'))    


# ## Test Model


sample = df_test.sample(1).to_dict(orient='records')
sample

X_sample = dv.fit_transform(sample)
X_sample


y_pred_sample = model.predict(X_sample)
y_pred_sample

