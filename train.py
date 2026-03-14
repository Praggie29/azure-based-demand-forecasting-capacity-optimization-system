# MILESTONE 1
#Data Collection and preparation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA

df=pd.read_csv("azure_cloud_usage_ml_dataset.csv")
 # print(df) -> displays the dataset
print(df.shape) #rows and columns
print(df.columns) #column names
print(df.info()) #data types and nulls
print(df.describe()) #statistics
print(df.head()) #output first 5 rows
print(df.isnull().sum()) #check for missing values
print(df.duplicated().sum()) #check for duplicates
df=df.drop_duplicates() #remove duplicates
print(df.shape) #new shape after removing duplicates
df.fillna(df.mode().iloc[0], inplace=True) #impute missing values with mode
print(df.isnull().sum()) #check again for missing values and this should show 0 for all columns
df['timestamp'] = pd.to_datetime(df['timestamp']) #convert Date to datetime
print(df.dtypes) #check data types
df=df.sort_values('timestamp') #sort by Date
print(df.head()) #check sorted data
df['usage_units']=df['usage_units'].fillna(df['usage_units'].median())
df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
df['day'] = df['timestamp'].dt.day
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['week_of_year'] = df['timestamp'].dt.isocalendar().week
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
holiday_months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] # Assuming holidays can occur in any month
df['is_holiday_month'] = df['month'].apply(lambda x: 1 if x in holiday_months else 0)
print(df.head()) #check new features
print(df.columns) #check all columns including new features

#MILESTONE 2
#Feature Engineering and Data Wrangling

df = df.set_index('timestamp')
# Seasonality Flags
df['day_of_week'] = df.index.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['month'] = df.index.month
df['quarter'] = df.index.quarter
df['is_month_start'] = df.index.is_month_start.astype(int)
df['is_month_end'] = df.index.is_month_end.astype(int)

df['usage_lag_1'] = df['usage_units'].shift(1)
df['usage_lag_7'] = df['usage_units'].shift(7)
df['usage_lag_30'] = df['usage_units'].shift(30)
df['rolling_mean_7'] = df['usage_units'].rolling(window=7).mean()
df['rolling_std_7'] = df['usage_units'].rolling(window=7).std()
df['rolling_mean_30'] = df['usage_units'].rolling(window=30).mean()
df['rolling_std_30'] = df['usage_units'].rolling(window=30).std()
df['usage_growth_rate'] = df['usage_units'].pct_change()
df['usage_spike'] = ( #spike detection
    df['usage_units'] > 
    (df['rolling_mean_7'] + 2 * df['rolling_std_7'])
).astype(int)
df['cumulative_mean_usage'] = df['usage_units'].expanding().mean()
df = df.dropna()
df = df.reset_index()
print("Final Shape:", df.shape)
print(df.head())
df.to_csv("azure_cloud_usage_ml_dataset.csv", index=False)
print(df.shape)
print(df.isnull().sum())
print(df.columns)


# MILESTONE 3
# Machine Learning Model Development


print(df.head())
print(df.info())

# Convert Timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')

# Target variable
target = "usage_units"

df['hour'] = df['timestamp'].dt.hour
df['day'] = df['timestamp'].dt.day
df['month'] = df['timestamp'].dt.month
df['weekday'] = df['timestamp'].dt.weekday

df['is_weekend'] = df['weekday'].isin([5,6]).astype(int)

# Lag features
df['lag_1'] = df[target].shift(1)
df['lag_24'] = df[target].shift(24)

# Rolling average
df['rolling_mean_24'] = df[target].rolling(24).mean()

df = df.dropna()

# Train Test Split
train_size = int(len(df) * 0.8)

train = df.iloc[:train_size]
test = df.iloc[train_size:]

X_train = train.drop([target, 'timestamp'], axis=1)
y_train = train[target]
X_test = test.drop([target, 'timestamp'], axis=1)
y_test = test[target]


# ARIMA MODEL
arima_model = ARIMA(y_train, order=(5,1,0))
arima_fit = arima_model.fit()
pred_arima = arima_fit.forecast(steps=len(y_test))


# RANDOM FOREST
rf_model = RandomForestRegressor(n_estimators=100)
rf_model.fit(X_train, y_train)
pred_rf = rf_model.predict(X_test)


# XGBOOST
xgb_model = XGBRegressor()
xgb_model.fit(X_train, y_train)
pred_xgb = xgb_model.predict(X_test)


# Evaluation Metrics
def evaluate(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    bias = np.mean(y_pred - y_true)

    print(name)
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("Bias:", bias)
    print("----------------------")


evaluate(y_test, pred_arima, "ARIMA")
evaluate(y_test, pred_rf, "Random Forest")
evaluate(y_test, pred_xgb, "XGBoost")


# Hyperparameter Tuning
params = {
    "n_estimators": [100, 200],
    "max_depth": [3, 6, 9],
    "learning_rate": [0.01, 0.1]
}

grid = GridSearchCV(
    XGBRegressor(),
    params,
    cv=3
)

grid.fit(X_train, y_train)
best_model = grid.best_estimator_
pred_best = best_model.predict(X_test)
evaluate(y_test, pred_best, "Tuned XGBoost")


# Backtesting
tscv = TimeSeriesSplit(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):

    X_tr = X_train.iloc[train_idx]
    X_val = X_train.iloc[val_idx]

    y_tr = y_train.iloc[train_idx]
    y_val = y_train.iloc[val_idx]

    model = XGBRegressor()

    model.fit(X_tr, y_tr)

    preds = model.predict(X_val)

    mae = mean_absolute_error(y_val, preds)

    print("Fold", fold+1, "MAE:", mae)



# Prediction Plot
plt.figure(figsize=(12,6))
plt.plot(y_test.values, label="Actual")
plt.plot(pred_best, label="Predicted")
plt.legend()
plt.title("Azure Cloud Demand Forecast")
plt.show()