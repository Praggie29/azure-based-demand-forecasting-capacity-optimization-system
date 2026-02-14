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