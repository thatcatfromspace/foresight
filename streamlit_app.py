import streamlit as st
import os
from datetime import datetime
from prediction.weather_prediction import WeatherPrediction
from storage.cassandra_client import CassandraClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Streamlit app configuration
st.set_page_config(page_title="Foresight Weather Prediction", layout="centered")
st.title("🌦️ Foresight Weather Prediction")
st.markdown("Enter a date and time to get weather predictions for temperature, precipitation, sunrise, and sunset.")

# Load weather data from Cassandra
@st.cache_data
def load_data():
    try:
        # Cassandra configuration from .env
        contact_points = [os.getenv("CASSANDRA_HOST", "localhost")]
        port = int(os.getenv("CASSANDRA_PORT", 9042))
        username = os.getenv("CASSANDRA_USERNAME", "cassandra")
        password = os.getenv("CASSANDRA_PASSWORD", "cassandra")
        keyspace = os.getenv("CASSANDRA_KEYSPACE", "foresight_keyspace")

        # Initialize Cassandra client
        cassandra_client = CassandraClient(contact_points, port, username, password, keyspace)

        # Retrieve data
        hourly_records = cassandra_client.retrieve_data('hourly_weather_data', 52537)
        daily_records = cassandra_client.retrieve_data('daily_weather_data', 2187)

        # Close Cassandra connection
        cassandra_client.close()
        
        return hourly_records, daily_records
    except Exception as e:
        st.error(f"Error loading data from Cassandra: {e}")
        return None, None

hourly_records, daily_records = load_data()

# Initialize WeatherPrediction instance
predictor = WeatherPrediction()

# User input for date and time
st.subheader("Select Date and Time")
col1, col2 = st.columns(2)
with col1:
    selected_date = st.date_input("Date", value=datetime(2025, 1, 1))
with col2:
    selected_time = st.time_input("Time", value=datetime(2025, 1, 1, 12, 0).time())

# Combine date and time into a datetime object
try:
    date_time = datetime.combine(selected_date, selected_time)
except Exception as e:
    st.error(f"Invalid date/time input: {e}")
    date_time = None

# Prediction button
if st.button("Get Weather Prediction"):
    if hourly_records is None or daily_records is None:
        st.error("Weather data not loaded. Please check Cassandra connection.")
    elif not hourly_records or not daily_records:
        st.error("Weather data is empty. Please verify Cassandra data.")
    elif date_time:
        with st.spinner("Generating predictions..."):
            try:
                # Make prediction with retrieved data
                predictor.make_prediction(hourly_records, daily_records)
                
                # Predict for user-selected date_time
                predictions = predictor.predict(date_time)
                
                # Display predictions
                st.subheader("Weather Predictions")
                st.metric(label="Temperature", value=f"{predictions['temperature_2m']:.1f} °C")
                st.metric(label="Precipitation", value=f"{predictions['precipitation']:.2f} mm")
                st.metric(label="Sunrise", value=predictions['sunrise_time'])
                st.metric(label="Sunset", value=predictions['sunset_time'])
                
            except Exception as e:
                st.error(f"Error generating predictions: {e}")
    else:
        st.warning("Please select a valid date and time.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ by Foresight Team | Powered by Streamlit")

# Close app button
if st.button("Close App"):
    st.stop()