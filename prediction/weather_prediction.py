from datetime import datetime
import pandas as pd
import numpy as np
from joblib import load
import os
from models.gbm import DataMerger
from transformations.transformations import Transformation

class WeatherPrediction:
    """
    A class to handle weather predictions including temperature, precipitation, sunrise, and sunset times.
    """
    def __init__(self):
        """Initialize with empty models and data."""
        self.models = {}
        self.data = pd.DataFrame()
        self.prediction_input = pd.DataFrame()

    def load_models(self):
        """
        Loads pre-trained models for temperature, precipitation, sunrise, and sunset times.
        """
        project_dir = os.path.dirname(os.path.abspath(__file__)) 
        models_dir = os.path.join(project_dir, '..', 'models/models')

        try:
            self.models = {
                'temperature_2m': load(os.path.join(models_dir, 'temperature_2m_model.joblib')),
                'precipitation': load(os.path.join(models_dir, 'precipitation_model.joblib')),
                'sunrise_numeric_sin': load(os.path.join(models_dir, 'sunrise_numeric_sin_model.joblib')),
                'sunrise_numeric_cos': load(os.path.join(models_dir, 'sunrise_numeric_cos_model.joblib')),
                'sunset_numeric_sin': load(os.path.join(models_dir, 'sunset_numeric_sin_model.joblib')),
                'sunset_numeric_cos': load(os.path.join(models_dir, 'sunset_numeric_cos_model.joblib')),
            }
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model file missing: {e}")
    
    def load_data(self, hourly_records, daily_records):
        """
        Merge hourly and daily records into a single dataset.
        """
        try:
            merger = DataMerger(hourly_records, daily_records)
            self.data = merger.merge_data()
        except Exception as e:
            raise Exception(f"Data merging failed: {e}")
        
        if self.data.empty:
            raise ValueError("Merged data is empty")

    def inverse_cyclic_encoding(self, sin_val, cos_val):
        """
        Converts sine and cosine values back to time (in hours).
        """
        angle = np.arctan2(sin_val, cos_val)
        if angle < 0:
            angle += 2 * np.pi
        time_in_hours = angle / (2 * np.pi) * 24
        return time_in_hours
    
    @staticmethod
    def get_season(month):
        """
        Returns the season based on the month.
        """
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        elif month in [9, 10, 11]:
            return 'fall'
            
    def calculate_seasonal_averages(self, input_year, features):
        """
        Calculates seasonal averages for specific weather features.
        """
        if 'timestamp' not in self.data.columns:
            return {}
        
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        previous_year = input_year - 1
        filtered_data = self.data[self.data['timestamp'].dt.year == previous_year]

        if filtered_data.empty:
            return {}

        filtered_data['season'] = filtered_data['timestamp'].dt.month.map(self.get_season)
        seasonal_avg = filtered_data.groupby('season')[features].mean().to_dict()
        
        reorganized_avg = {}
        for feature, season_avg in seasonal_avg.items():
            for season, avg in season_avg.items():
                if season not in reorganized_avg:
                    reorganized_avg[season] = {}
                reorganized_avg[season][feature] = avg

        return reorganized_avg
        
    def prepare_input(self, date_time):
        """
        Prepares input features for the prediction models.

        Args:
            date_time (datetime): The datetime for which the weather prediction should be made.

        Returns:
            DataFrame: The prepared input features, including cyclic encoding and seasonal averages if applicable.
        """
        self.prediction_input = pd.DataFrame()
        self.prediction_input['year'] = [date_time.year]
        self.prediction_input['day_of_year'] = [date_time.day]
        self.prediction_input['hour'] = [date_time.hour]

        trans = Transformation(self.prediction_input)
        features_cycle = [{'feature': 'hour', 'cycle': 24}, {'feature': 'day_of_year', 'cycle': 365}]
        self.prediction_input = trans.add_cyclic_encoding(features_cycle)

        target_columns = ['temperature_2m', 'precipitation', 'sunrise_numeric_sin', 'sunrise_numeric_cos', 'sunset_numeric_sin', 'sunset_numeric_cos']
        date_time_col = ['timestamp', 'date', 'sunrise', 'sunset', 'year', 'day_of_year', 'hour']
        
        features = [col for col in self.data.columns if col not in target_columns + date_time_col]

        self.prediction_input, features_seasonal_avg = trans.fill_missing_features(self.data, features, [24, 48, 72], date_time)

        if features_seasonal_avg:
            seasonal_avg = self.calculate_seasonal_averages(date_time.year, features_seasonal_avg)
            month = date_time.month
            current_season = self.get_season(month)

            if current_season not in seasonal_avg:
                raise KeyError(f"Seasonal averages for the season '{current_season}' are missing. Available keys: {list(seasonal_avg.keys())}")

            for feature, avg in seasonal_avg[current_season].items():
                self.prediction_input[feature] = avg

        return self.prediction_input

    def predict(self, date_time):
        """
        Predicts weather parameters for a given datetime.
        """
        try:
            self.prediction_input = self.prepare_input(date_time)
            if self.prediction_input.empty:
                raise ValueError("Prediction input is empty")
            
            temperature = self.models['temperature_2m'].predict(self.prediction_input)[0]
            precipitation = self.models['precipitation'].predict(self.prediction_input)[0]

            sunrise_sin = self.models['sunrise_numeric_sin'].predict(self.prediction_input)[0]
            sunrise_cos = self.models['sunrise_numeric_cos'].predict(self.prediction_input)[0]
            sunset_sin = self.models['sunset_numeric_sin'].predict(self.prediction_input)[0]
            sunset_cos = self.models['sunset_numeric_cos'].predict(self.prediction_input)[0]

            sunrise_time = self.inverse_cyclic_encoding(sunrise_sin, sunrise_cos)
            sunset_time = self.inverse_cyclic_encoding(sunset_sin, sunset_cos)

            sunrise_time_str = f"{int(sunrise_time):02d}:{int((sunrise_time % 1) * 60):02d}"
            sunset_time_str = f"{int(sunset_time):02d}:{int((sunset_time % 1) * 60):02d}"

            return {
                'temperature_2m': temperature,
                'precipitation': precipitation,
                'sunrise_time': sunrise_time_str,
                'sunset_time': sunset_time_str
            }
        except Exception as e:
            raise Exception(f"Prediction failed: {e}")
    
    def make_prediction(self, hourly_records, daily_records):
        """
        Loads models and data, then makes weather predictions.
        """
        try:
            self.load_models()
            self.load_data(hourly_records, daily_records)
            date_time = datetime(2025, 1, 1, 12, 0)
            
            predictions = self.predict(date_time)

            print("Predictions:")
            print(f"Temperature: {predictions['temperature_2m']}°C")
            print(f"Precipitation: {predictions['precipitation']} mm")
            print(f"Sunrise: {predictions['sunrise_time']}")
            print(f"Sunset: {predictions['sunset_time']}")
        except Exception as e:
            raise Exception(f"Make prediction failed: {e}")