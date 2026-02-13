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
