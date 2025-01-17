## ForeSight

A full-fledged big data model to predict and analyze temperatures.

## Installing and running 

We need `Kafka` to process data and `Cassandra` as the database. Both of them are installed locally using `Docker`. 

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

