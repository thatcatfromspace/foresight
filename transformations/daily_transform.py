from transformations.transformations import Transformation
import pandas as pd
import json

class DailyTransform:
    def __init__(self, data_path):
        with open(data_path) as f:
            sample = json.load(f)

        records = []
        for date, entries in sample.items():
            for entry in entries:
                record = entry.copy() 
                records.append(record)

        self.df = pd.DataFrame(records)

        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df['year'] = self.df['date'].dt.year
        self.df['day_of_year'] = self.df['date'].dt.dayofyear

        self.trans = Transformation(self.df)

    def convert_time_to_num(self, df, features):
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


    def daily_transformed_data(self):   
        self.df = self.convert_time_to_num(self.df, ['sunrise', 'sunset'])
        features_cycle = [{'feature': 'day_of_year', 'cycle': 365},{'feature': 'sunrise_numeric', 'cycle': 24}, {'feature': 'sunset_numeric', 'cycle': 24}]

        self.df = self.trans.add_cyclic_encoding(features_cycle)
        self.df.drop(columns=['sunrise_numeric', 'sunset_numeric'], inplace=True)

        columns_to_lag = ['temperature_2m_max', 'temperature_2m_min', 'precipitation_sum']
        lag_day = [1, 3]

        self.df = self.trans.add_lag_features(columns_to_lag, lag_day,'date')

        self.df = self.df.map(lambda x: x.strftime("%Y-%m-%d") if isinstance(x, pd.Timestamp) else x)

        records_list = self.df.to_dict(orient="records")

        return records_list

# Sample usage
if __name__ == '__main__':
    data_path = '../daily_weather_data.json'
    daily_transform = DailyTransform(data_path)
    records = daily_transform.daily_transformed_data()
    print(records)
