from kafka import KafkaProducer
from ingestion.open_meteo_client import OpenMeteoClient  
import json
from openmeteopy.hourly import HourlyHistorical
from openmeteopy.daily import DailyHistorical
import os
import logging

class WeatherDataProducer:
    def __init__(self, kafka_broker: str, hourly_topic: str, daily_topic: str):
        """
        Initialize the Kafka producer and set the topics.

        Args:
            kafka_broker (str): Kafka broker address.
            hourly_topic (str): Kafka topic to send hourly data to.
            daily_topic (str): Kafka topic to send daily data to.
        """
        self.producer = KafkaProducer(
            bootstrap_servers=[kafka_broker],
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8'),
            compression_type='gzip',
            batch_size=16384, # 16 KB batch size
            linger_ms=5,  # Time to wait before sending a batch
            buffer_memory=33554432 # 32 MB buffer size
        )
        self.hourly_topic = hourly_topic
        self.daily_topic = daily_topic

    def send_weather_data(self, latitude, longitude, start_date, end_date):
        """
        Fetch weather data using OpenMeteoClient and send it to Kafka.

        Args:
            latitude (float): Latitude of the location.
            longitude (float): Longitude of the location.
            start_date (str): Start date in YYYY-MM-DD format.
            end_date (str): End date in YYYY-MM-DD format.
        """
        try:
            client = OpenMeteoClient(latitude=latitude, longitude=longitude, start_date=start_date, end_date=end_date)

            client.config_daily_params(
                DailyHistorical()
                .temperature_2m_max()
                .temperature_2m_min()
                .precipitation_sum()
                .sunrise()
                .sunset()
            )

            client.config_hourly_params(
                HourlyHistorical()
                .precipitation()
                .cloudcover()
                .temperature_2m()
                .windspeed_10m()
                .dewpoint_2m()
                .winddirection_10m()
                .relativehumidity_2m()
            )

            weather_data = client.get_weather_data()

            hourly_time = weather_data.get('hourly', {}).get('time', [])
            if not hourly_time:
                logging.info("No hourly data available.")
            else:
                batch_size = 24  # Send hourly data in batches of 24 hours 
                hourly_data_batch = []

                for idx, timestamp in enumerate(hourly_time):
                    hourly_data = {
                        'timestamp': timestamp,
                        'precipitation': weather_data['hourly']['precipitation'][idx],
                        'cloudcover': weather_data['hourly']['cloudcover'][idx],
                        'temperature_2m': weather_data['hourly']['temperature_2m'][idx],
                        'windspeed_10m': weather_data['hourly']['windspeed_10m'][idx],
                        'dewpoint_2m': weather_data['hourly']['dewpoint_2m'][idx],
                        'winddirection_10m': weather_data['hourly']['winddirection_10m'][idx],
                        'relativehumidity_2m': weather_data['hourly']['relativehumidity_2m'][idx]
                    }
                    hourly_data_batch.append(hourly_data)

                    if len(hourly_data_batch) == batch_size:
                        self.producer.send(self.hourly_topic, key=f"{timestamp[:10]}", value=hourly_data_batch)
                        hourly_data_batch = []  # Reset batch

                # Send any remaining data in the batch
                if hourly_data_batch:
                    self.producer.send(self.hourly_topic, key=f"{hourly_time[-1][:10]}", value=hourly_data_batch)

            # Handle daily data separately because only one entry is available per day
            daily_time = weather_data.get('daily', {}).get('time', [])
            if not daily_time:
                logging.info("No daily data available.")
            else:
                for idx, date in enumerate(daily_time):
                    daily_data = {
                        'sunrise': weather_data['daily']['sunrise'][idx],
                        'sunset': weather_data['daily']['sunset'][idx],
                        'temperature_2m_max': weather_data['daily']['temperature_2m_max'][idx],
                        'temperature_2m_min': weather_data['daily']['temperature_2m_min'][idx],
                        'precipitation_sum': weather_data['daily']['precipitation_sum'][idx]
                    }
                    self.producer.send(self.daily_topic, key=date, value=daily_data)

            logging.info("Weather data successfully sent to Kafka.")
        except Exception as e:
            logging.error(f"Error in kafka_producer.py: {e}")

    def close(self):
        """Close the Kafka producer."""
        self.producer.close()

if __name__ == "__main__":
    kafka_broker = os.getenv("KAFKA_BROKER") 
    hourly_topic = os.getenv("KAFKA_HOURLY_TOPIC")
    daily_topic = os.getenv("KAFKA_DAILY_TOPIC")

    latitude = 37.7749  
    longitude = -122.4194
    start_date = "2022-01-01"
    end_date = "2022-01-07"

    producer = WeatherDataProducer(kafka_broker, hourly_topic, daily_topic)
    producer.send_weather_data(latitude, longitude, start_date, end_date)
    producer.close()
