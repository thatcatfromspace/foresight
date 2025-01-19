import os
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.query import SimpleStatement
from cassandra.policies import DCAwareRoundRobinPolicy
import logging
import json

class CassandraClient:
    def __init__(self, contact_points, port, username=None, password=None, keyspace=None):
        """
        Initialize the CassandraClient.

        Args:
            contact_points (list): List of Cassandra nodes.
            port (int): Port number for Cassandra connection.
            username (str): Username for authentication (optional).
            password (str): Password for authentication (optional).
            keyspace (str): Keyspace to use (optional).
        """
        
        self.keyspace = keyspace
        
        auth_provider = PlainTextAuthProvider(username, password) if username and password else None
        self.cluster = Cluster(contact_points, port=port, auth_provider=auth_provider, load_balancing_policy=DCAwareRoundRobinPolicy(local_dc='datacenter1'))
        self.session = self.cluster.connect()
        
        self.schema_path = os.getenv("CASSANDRA_SCHEMA", "schema.cql") 
        
        self.session.execute("CREATE KEYSPACE IF NOT EXISTS foresight_keyspace WITH REPLICATION = {'class': 'SimpleStrategy', 'replication_factor': 1}")  
        self.session.set_keyspace(keyspace)

        self.execute_schema(os.getenv("CASSANDRA_SCHEMA", "schema.cql"))

    def execute_schema(self, schema_path):
        """
        Execute a CQL schema file to set up tables or a keyspace.

        Args:
            schema_path (str): Path to the .cql schema file.
        """
        try:
            with open(self.schema_path, 'r') as file:
                schema = file.read()
                statements = schema.split(';') 
                for statement in statements:
                    if statement.strip():
                        self.session.execute(SimpleStatement(statement))
            logging.info(f"Schema executed successfully from {schema_path}.")
        except Exception as e:
            logging.error(f"Error executing schema: {e}")

    def insert_hourly_data(self, table, data):
        """
        Insert hourly weather data into a table.

        Args:
            table (str): Table name.
            data (dict): Dictionary containing the data to insert.
        """
        try:
            query = f"INSERT INTO {table} JSON %s"
            self.session.execute(query, (json.dumps(data),))
        except Exception as e:
            logging.error(f"Error inserting data: {e}")
            

    def insert_daily_data(self, table, data):
        """
        Insert daily weather data into a table.

        Args:
            table (str): Table name.
            data (dict): Dictionary containing the data to insert.
        """
        try:
            query = f"INSERT INTO {table} JSON %s"
            self.session.execute(query, (json.dumps(data),))
        except Exception as e:
            logging.error(f"Error inserting data: {e}")

    def retrieve_data(self, table, limit=10):
        """
        Retrieve data from a table with an optional limit.

        Args:
            table (str): Table name.
            limit (int): Number of rows to retrieve (default: 10).

        Returns:
            list: Retrieved rows.
        """
        try:
            query = f"SELECT * FROM {table} LIMIT {limit}"
            rows = self.session.execute(query)
            return [row for row in rows]
        except Exception as e:
            logging.error(f"Error retrieving data: {e}")
            return []

    def close(self):
        """Close the Cassandra session and cluster connection."""
        self.cluster.shutdown()
        logging.info("Cassandra connection closed.")

# Sample usage 
if __name__ == "__main__":
    contact_points = [os.getenv("CASSANDRA_HOST", "localhost")]  
    port = int(os.getenv("CASSANDRA_PORT", 9042))
    username = os.getenv("CASSANDRA_USERNAME", "cassandra")
    password = os.getenv("CASSANDRA_PASSWORD", "cassandra")
    keyspace = os.getenv("CASSANDRA_KEYSPACE", "foresight_keyspace")

    cassandra_client = CassandraClient(contact_points, port, username, password, keyspace)
    
    cassandra_client.execute_schema(os.getenv("CASSANDRA_SCHEMA", "schema.cql"))

    hourly_data = {
        "timestamp": "2025-01-18T12:00:00Z",
        "precipitation": 0.0,
        "cloudcover": 20,
        "temperature_2m": 15.3,
        "windspeed_10m": 5.2,
        "dewpoint_2m": 10.5,
        "winddirection_10m": 180,
        "relativehumidity_2m": 80,
        "year": 2025,
        "hour": 12,
        "day_of_year": 18,
        "hour_sin": 0.5,
        "hour_cos": 0.5,
        "day_of_year_sin": 0.1,
        "day_of_year_cos": 0.9,
        "precipitation_lag_6": 0.0,
        "precipitation_lag_24": 0.0,
        "precipitation_lag_48": 0.0,
        "cloudcover_lag_6": 15,
        "cloudcover_lag_24": 30,
        "cloudcover_lag_48": 25,
        "temperature_2m_lag_6": 15.0,
        "temperature_2m_lag_24": 14.8,
        "temperature_2m_lag_48": 14.5,
        "windspeed_10m_lag_6": 5.0,
        "windspeed_10m_lag_24": 4.8,
        "windspeed_10m_lag_48": 4.5,
        "dewpoint_2m_lag_6": 10.0,
        "dewpoint_2m_lag_24": 9.8,
        "dewpoint_2m_lag_48": 9.5,
        "winddirection_10m_lag_6": 190,
        "winddirection_10m_lag_24": 185,
        "winddirection_10m_lag_48": 180,
        "relativehumidity_2m_lag_6": 82,
        "relativehumidity_2m_lag_24": 80,
        "relativehumidity_2m_lag_48": 78,
        "precipitation_rolling_6h": 0.0,
        "cloudcover_rolling_6h": 22,
        "temperature_2m_rolling_6h": 15.1,
        "windspeed_10m_rolling_6h": 5.1,
        "dewpoint_2m_rolling_6h": 10.3,
        "relativehumidity_2m_rolling_6h": 81
    }

    table_name = "hourly_weather_data" 
    cassandra_client.insert_hourly_data(table_name, hourly_data)

    retrieved_data = cassandra_client.retrieve_data(table_name)
    print("Retrieved Data:", retrieved_data[0])

    cassandra_client.close()
