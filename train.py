# MILESTONE 1
#Data Collection and preparation

import pandas as pd
import numpy as np

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