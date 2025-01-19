from transformations import Transformation
import pandas as pd
import json
import numpy as np

data_sample='/Users/noorfathima/Downloads/hourly_weather_data.json'
with open(data_sample) as f:
    sample = json.load(f)

records = []
for date, entries in sample.items():
    for entry in entries:
        record = entry.copy() 
        records.append(record)

df = pd.DataFrame(records)

df['timestamp'] = pd.to_datetime(df['timestamp'])
df['year'] = df['timestamp'].dt.year  
df['hour'] = df['timestamp'].dt.hour  
df['day_of_year'] = df['timestamp'].dt.dayofyear 

trans = Transformation(df)

features_cycle = [{'feature': 'hour', 'cycle': 24}, {'feature': 'day_of_year', 'cycle': 365}]
df = trans.add_cyclic_encoding(features_cycle)

# features_to_standardize = ['temperature_2m', 'dewpoint_2m', 'windspeed_10m']
# df = trans.standardize_by_year(features_to_standardize)

# df = trans.log_transform_precipitation('precipitation')

# features_to_normalize = ['relativehumidity_2m', 'cloudcover', 'precipitation_log']
# df = trans.normalize_data(features_to_normalize)

columns_to_lag = [
    'precipitation', 'cloudcover', 'temperature_2m', 
    'windspeed_10m', 'dewpoint_2m', 'winddirection_10m', 
    'relativehumidity_2m'
]
lag_hours = [6, 24, 48]
df = trans.add_lag_features(columns_to_lag, lag_hours,'timestamp')

# lag_columns_to_standardize=['temperature_2m_lag_6', 'dewpoint_2m_lag_6', 'windspeed_10m_lag_6',
#                             'temperature_2m_lag_24', 'dewpoint_2m_lag_24', 'windspeed_10m_lag_24',
#                             'temperature_2m_lag_48', 'dewpoint_2m_lag_48', 'windspeed_10m_lag_48']
# df = trans.standardize_by_year(lag_columns_to_standardize)

# df = trans.log_transform_precipitation('precipitation_lag_6')
# df = trans.log_transform_precipitation('precipitation_lag_24')
# df = trans.log_transform_precipitation('precipitation_lag_48')

# lag_columns_to_normalize=['relativehumidity_2m_lag_6', 'cloudcover_lag_6', 'precipitation_lag_6_log',
#  'relativehumidity_2m_lag_24', 'cloudcover_lag_24', 'precipitation_lag_24_log',
#  'relativehumidity_2m_lag_48', 'cloudcover_lag_48', 'precipitation_lag_48_log']
# df = trans.normalize_data(lag_columns_to_normalize)

features_to_rolling_mean = ['precipitation', 'cloudcover', 'temperature_2m', 
                            'windspeed_10m', 'dewpoint_2m', 'relativehumidity_2m']
df = trans.add_rolling_mean(features_to_rolling_mean, window=6)

# rolling_features_to_standardize = ['temperature_2m_rolling_6h', 'dewpoint_2m_rolling_6h', 'windspeed_10m_rolling_6h']
# df = trans.standardize_by_year(rolling_features_to_standardize)

# df = trans.log_transform_precipitation('precipitation_rolling_6h')

# rolling_features_to_normalize = ['relativehumidity_2m_rolling_6h', 'cloudcover_rolling_6h', 'precipitation_rolling_6h_log']
# df = trans.normalize_data(rolling_features_to_normalize)

print(df)

df.to_csv('/Users/noorfathima/Documents/college/code/weather/final_transformed_data.csv', index=False)




