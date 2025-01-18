from kafka import KafkaConsumer
import json

def consume_messages(topic):
    # Create a Kafka consumer
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=['localhost:9092'],  # Replace with your Kafka broker address
        group_id='weather_data_consumer',  # Consumer group ID
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        key_deserializer=lambda k: k.decode('utf-8')
    )

    print(f"Consuming messages from topic: {topic}")
    with open('hourly_weather_data.json', 'w+') as f:
        for message in consumer:
            print(f"Consumed message: {message.value}")
            json.dump(message.value, f)
            f.write(',')

if __name__ == "__main__":
    response = consume_messages('hourly_weather_data')  # For hourly data
    # consume_messages('daily_weather_data')  # Uncomment for daily data
