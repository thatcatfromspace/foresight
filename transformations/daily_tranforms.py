from transformations import Transformation
import pandas as pd
import json
import numpy as np

data_path='/Users/noorfathima/Downloads/daily_weather_data.json'
with open(data_path) as f:
    sample = json.load(f)

records = []
for date, entries in sample.items():
    for entry in entries:
        record = entry.copy() 
        records.append(record)

df = pd.DataFrame(records)

df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['day_of_year'] = df['date'].dt.dayofyear

trans = Transformation(df)
def convert_time_to_num(df,features):
    """
    Converts time-related features to numeric values representing hours.

    Args:
    df : pandas.DataFrame
        The DataFrame containing the time-related features to convert.
    
    features : list of str
        A list of column names that contain time-related data (e.g., timestamps).
        These columns will be converted to numeric values (in hours).

    Returns:
    df : The DataFrame with the new numeric columns added. 
    """
    for feature in features:
     df[f'{feature}_numeric'] = pd.to_datetime(df[feature]).dt.hour + pd.to_datetime(df[feature]).dt.minute / 60
    return df

df = convert_time_to_num(df, ['sunrise', 'sunset'])

features_cycle = [{'feature': 'day_of_year', 'cycle': 365},{'feature': 'sunrise_numeric', 'cycle': 24}, {'feature': 'sunset_numeric', 'cycle': 24}]

df = trans.add_cyclic_encoding(features_cycle)
df.drop(columns=['sunrise_numeric', 'sunset_numeric'], inplace=True)

columns_to_lag = ['temperature_2m_max', 'temperature_2m_min', 'precipitation_sum']
lag_day = [1, 3]
df = trans.add_lag_features(columns_to_lag, lag_day,'date')

df.to_json('/Users/noorfathima/Documents/college/project/foresight/transformations/daily_transformed.json', orient="records", lines=True, indent=4)
