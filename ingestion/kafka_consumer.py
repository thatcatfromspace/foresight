from kafka import KafkaConsumer
import json
import os
import logging

class WeatherDataConsumer:
    def __init__(self, kafka_broker: str, hourly_topic: str, daily_topic: str, hourly_filename: str, daily_filename: str):
        """
        Initialize the Kafka consumer.

        Args:
            kafka_broker (str): Kafka broker address.
            hourly_topic (str): Kafka topic to consume hourly data from.
            daily_topic (str): Kafka topic to consume daily data from.
            hourly_filename (str): Filename to store hourly data.
            daily_filename (str): Filename to store daily data.
        """
        self.consumer = KafkaConsumer(
            hourly_topic,
            daily_topic,
            bootstrap_servers=[kafka_broker],
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),
            key_deserializer=lambda k: k.decode('utf-8'),
            consumer_timeout_ms=5000,
        )
        
        self.hourly_filename = hourly_filename
        self.daily_filename = daily_filename

        self.hourly_topic = hourly_topic
        self.daily_topic = daily_topic

        self.hourly_data = []
        self.daily_data = []

    def receive_weather_data(self):
        """
        Consume data from Kafka and save it to separate files for hourly and daily data.
        """
        try:
            for message in self.consumer:
                topic = message.topic
                key = message.key
                value = message.value

                if topic == self.daily_topic:
                    self.daily_data.append({
                        'date': key,
                        'temperature_2m_max': value['temperature_2m_max'],
                        'temperature_2m_min': value['temperature_2m_min'],
                        'precipitation_sum': value['precipitation_sum'],
                        'sunrise': value['sunrise'],
                        'sunset': value['sunset']
                    })
                    logging.info(f"Received data from topic: {self.daily_topic}, key: {key}")
                    self.save_to_file(self.daily_filename, self.daily_data, 'daily')
                
                elif topic == self.hourly_topic:  
                    for hourly_record in value:
                        self.hourly_data.append({
                            'timestamp': hourly_record['timestamp'],
                            'precipitation': hourly_record['precipitation'],
                            'cloudcover': hourly_record['cloudcover'],
                            'temperature_2m': hourly_record['temperature_2m'],
                            'windspeed_10m': hourly_record['windspeed_10m'],
                            'dewpoint_2m': hourly_record['dewpoint_2m'],
                            'winddirection_10m': hourly_record['winddirection_10m'],
                            'relativehumidity_2m': hourly_record['relativehumidity_2m']
                        })
                    logging.info(f"Received hourly data batch from topic: {self.hourly_topic}, key: {key}")
                    self.save_to_file(self.hourly_filename, self.hourly_data, 'hourly')

            
            self.save_to_file(self.hourly_filename, self.hourly_data, 'hourly')
            self.save_to_file(self.daily_filename, self.daily_data, 'daily')

        except Exception as e:
            logging.info(f"Error: {e}")

    def save_to_file(self, filename, data, interval):
        """Save the aggregated data to a JSON file."""
        try:
            with open(filename, 'w+') as file:
                # Wrap data in a parent object 
                json.dump({interval: data}, file, indent=2)
        except Exception as e:
            logging.error(f"Error saving data to file in kafka_consumer.py: {e}")

    def close(self):
        """Close the Kafka consumer."""
        self.consumer.close()

if __name__ == "__main__":
    kafka_broker = os.getenv("KAFKA_BROKER") 
    hourly_topic = os.getenv("KAFKA_HOURLY_TOPIC")
    daily_topic = os.getenv("KAFKA_DAILY_TOPIC")

    hourly_filename = "hourly_weather_data.json"
    daily_filename = "daily_weather_data.json"

    consumer = WeatherDataConsumer(kafka_broker, hourly_topic, daily_topic, hourly_filename, daily_filename)
    consumer.receive_weather_data()
    consumer.close()
