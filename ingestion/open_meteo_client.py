from openmeteopy import OpenMeteo as Client
from openmeteopy.hourly import HourlyHistorical
from openmeteopy.daily import DailyHistorical
from openmeteopy.options import HistoricalOptions

class OpenMeteoClient:
    def __init__(self, latitude: float, longitude: float, start_date: str, end_date: str) -> None:
        """
        OpenMeteoClient class to get weather data from OpenMeteo API. Hourly and daily parameters
        must be explicitly configured before fetching data from the API else only metadata will be received. 
        Sample usage included in the class definition.

        Args:
            latitude (float): Latitude of the location.
            longitude (float): Longitude of the location.
            start_date (str): Start date in YYYY-MM-DD format.
            end_date (str): End date in YYYY-MM-DD format.
        """
        self.hourly = HourlyHistorical()
        self.daily = DailyHistorical()
        self.options = HistoricalOptions(latitude=latitude, longitude=longitude, start_date=start_date, end_date=end_date)

    def config_daily_params(self, daily_params: DailyHistorical) -> None:
        self.daily = daily_params

    def config_hourly_params(self, hourly_params: HourlyHistorical) -> None:
        self.hourly = hourly_params

    def get_weather_data(self) -> dict:
        self.client = Client(self.options, self.hourly, self.daily)
        response = self.client.get_dict()
        return response
        
    def save_weather_data(self, filepath: str) -> str:
        self.client = Client(self.options, self.hourly, self.daily)
        self.client.save_json(filepath)
        return "Weather data saved successfully."


# Sample usage
# ----------------
# longitude = 76.98
# latitude =  11.01
# start_date = "2021-05-01"
# end_date = "2021-05-02"

# client = OpenMeteoClient(latitude=latitude, longitude=longitude, start_date=start_date, end_date=end_date)
# client.config_daily_params(client.daily.temperature_2m_max().temperature_2m_min().precipitation_sum())
# client.config_hourly_params(client.hourly.precipitation().cloudcover().temperature_2m().windspeed_10m().dewpoint_2m().winddirection_10m().relativehumidity_2m())

# response = client.save_weather_data("weather.json")

