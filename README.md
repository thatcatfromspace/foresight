## ForeSight

"Sooooo... The weather?"

A full-fledged big data model to predict and analyze temperatures.

## Installing and running 

We need `Kafka` to process data and `Cassandra` as the database. Both of them are installed locally using `Docker`. 

To run a local Kafka server, make sure you have Docker installed, and then run the command.

```bash
cd kafka
docker compose up -d
```

This will create a network running Zookeeper and Kafka with the required topics: `hourly_weather_data` and `daily_weather_data`.

To run the Python application, first create a `venv`.

```bash
# Windows
py -m venv venv

# Linux/Mac
python3 -m venv venv
```

Move to the directory and activate the virtual environment.

```bash
# Windows
venv\Scripts\Activate

# Linux/Mac
source venv/bin/activate
```

Install the requirements.

```bash
pip install -r requirements.txt
```


NOTE: `openmeteopy` is not available on PyPI yet. It must be installed from GitHub using the following command: 

```sh
pip install git+https://github.com/m0rp43us/openmeteopy
```