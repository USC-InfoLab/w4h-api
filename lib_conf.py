# Sensor API
HOST = '127.0.0.1'
URL_PATH = 'fetchdata'
PORT = '9977'
DATA_TYPE = 'DATABASE' # 'DATABASE' or 'CSV'
DATASET = '<dataset_path>' # Path of the CSV dataset. Leave empty if you want to stream from database table

# # Database Config

DB_TABLE = 'heart_rates'
DB_USER_TABLE = 'subjects'
DB_CALORIES_TABLE = 'calories'
DB_COORDINATES_TABLE = 'locations'

# After each TIMEOUT (seconds), BATCH number of data entries is sent by the API
# Start time is the time recorded data stream is simulated to start from
DATE_TIME_COL = 'timestamp'
BATCH = 1
TIMEOUT = 5
START_TIME = '2016-08-01 12:00:00'
END_TIME = '2016-08-14 23:59:59'



db_config_path = 'conf/db_config.yaml'

# DEFAULT_START_DATE = date.today()
ACTIVITIES_REAL_INTERVAL = 15
ALERT_TIMEOUT = 60
DEFAULT_WINDOW_SIZE = 60
DEFAULT_MIN_HRATE = 60
DEFAULT_MAX_HRATE = 115

# Define the USC location as a latitude and longitude
USC_CENTER_Y = 34.0224
USC_CENTER_X = -118.2851

