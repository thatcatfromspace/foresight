from dotenv import load_dotenv
from ingestion.kafka_producer import WeatherDataProducer
from ingestion.kafka_consumer import WeatherDataConsumer
from storage.cassandra_client import CassandraClient
import os
import threading
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", handlers=[logging.FileHandler("logs/app.log"), logging.StreamHandler()])

load_dotenv()

kafka_broker = os.getenv("KAFKA_BROKER")
hourly_topic = os.getenv("KAFKA_HOURLY_TOPIC")
daily_topic = os.getenv("KAFKA_DAILY_TOPIC")

def run_producer():
    """
    Initializes and runs the Kafka producer to send weather data.
    """

    latitude = os.getenv("LATITUDE")
    longitude = os.getenv("LONGITUDE")
    start_date = os.getenv("START_DATE")
    end_date = os.getenv("END_DATE")

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
    # Kafka broker
    producer_thread = threading.Thread(target=run_producer, name="ProducerThread")
    consumer_thread = threading.Thread(target=run_consumer, name="ConsumerThread", daemon=True)

    producer_thread.start()
    consumer_thread.start()

    producer_thread.join()
    consumer_thread.join()

    # Insert into database 
    contact_points = [os.getenv("CASSANDRA_HOST", "localhost")]  
    port = int(os.getenv("CASSANDRA_PORT", 9042))
    username = os.getenv("CASSANDRA_USERNAME", "cassandra")
    password = os.getenv("CASSANDRA_PASSWORD", "cassandra")
    keyspace = os.getenv("CASSANDRA_KEYSPACE", "foresight_keyspace")
    cassandra_client = CassandraClient(contact_points, port, username, password, keyspace)