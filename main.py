from dotenv import load_dotenv
from ingestion.kafka_producer import WeatherDataProducer
from ingestion.kafka_consumer import WeatherDataConsumer
import os
import threading
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", handlers=[logging.FileHandler("app.log"), logging.StreamHandler()])

load_dotenv()

kafka_broker = os.getenv("KAFKA_BROKER")
hourly_topic = os.getenv("KAFKA_HOURLY_TOPIC")
daily_topic = os.getenv("KAFKA_DAILY_TOPIC")

def run_producer():
    """
    Initializes and runs the Kafka producer to send weather data.
    """

    latitude = 37.7749  
    longitude = -122.4194
    start_date = "2022-01-01"
    end_date = "2022-01-07"

    producer = WeatherDataProducer(kafka_broker, hourly_topic, daily_topic)
    producer.send_weather_data(latitude, longitude, start_date, end_date)
    producer.close()
    logging.info("Producer has finished sending data.")

def run_consumer():
    """
    Initializes and runs the Kafka consumer to process weather data.
    """

    hourly_filename = "hourly_weather_data.json"
    daily_filename = "daily_weather_data.json"

    consumer = WeatherDataConsumer(kafka_broker, hourly_topic, daily_topic, hourly_filename, daily_filename)
    consumer.receive_weather_data()
    consumer.close()
    logging.info("Consumer has finished processing data.")

if __name__ == "__main__":
    
    producer_thread = threading.Thread(target=run_producer, name="ProducerThread")
    consumer_thread = threading.Thread(target=run_consumer, name="ConsumerThread", daemon=True)

    producer_thread.start()
    consumer_thread.start()

    producer_thread.join()
    consumer_thread.join()
