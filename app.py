from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from joblib import load
import pandas as pd
import numpy as np
import os

app = FastAPI()

class WeatherPredictionInput(BaseModel):
    date_time: datetime

class WeatherPrediction:
    def __init__(self):
        self.models = {}
        self.data = pd.DataFrame()
        self.prediction_input = pd.DataFrame()

    def load_models(self):
        project_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(project_dir, 'models/models')

        self.models = {
            'temperature_2m': load(os.path.join(models_dir, 'temperature_2m_model.joblib')),
            'precipitation': load(os.path.join(models_dir, 'precipitation_model.joblib')),
            'sunrise_numeric_sin': load(os.path.join(models_dir, 'sunrise_numeric_sin_model.joblib')),
            'sunrise_numeric_cos': load(os.path.join(models_dir, 'sunrise_numeric_cos_model.joblib')),
            'sunset_numeric_sin': load(os.path.join(models_dir, 'sunset_numeric_sin_model.joblib')),
            'sunset_numeric_cos': load(os.path.join(models_dir, 'sunset_numeric_cos_model.joblib')),
        }

    def inverse_cyclic_encoding(self, sin_val, cos_val):
        angle = np.arctan2(sin_val, cos_val)
        if angle < 0:
            angle += 2 * np.pi
        time_in_hours = angle / (2 * np.pi) * 24
        return time_in_hours

    def prepare_input(self, date_time):
        self.prediction_input['year'] = date_time.year
        self.prediction_input['day_of_year'] = date_time.timetuple().tm_yday
        self.prediction_input['hour'] = date_time.hour
        # Add additional feature preparation as needed
        return self.prediction_input

    def predict(self, date_time):
        self.prediction_input = self.prepare_input(date_time)
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

weather_predictor = WeatherPrediction()
weather_predictor.load_models()

@app.post("/predict")
async def predict_weather(input_data: WeatherPredictionInput):
    try:
        prediction = weather_predictor.predict(input_data.date_time)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
