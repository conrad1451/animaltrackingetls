import os
from flask import Flask, Response, request, send_file
# import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

# Libraries needed (pandas is not standard and must be installed in Python)
import requests
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

FROST_API_CLIENTID = os.getenv("FROST_API_CLIENTID", 'YOUR_API_KEY')

# Insert your own client ID here
client_id = FROST_API_CLIENTID

# Define endpoint and parameters
FROST_BASE_URL = 'https://frost.met.no/'
parameters = {
    'sources': 'SN18700,SN90450',
    'elements': 'mean(air_temperature P1D),sum(precipitation_amount P1D),mean(wind_speed P1D)',
    'referencetime': '2010-04-01/2010-04-03',
}


def get_full_table():
    # This will return a Dataframe with all of the observations in a table format
    df = pd.DataFrame()
    for i in range(len(data)):
        row = pd.DataFrame(data[i]['observations'])
        row['referenceTime'] = data[i]['referenceTime']
        row['sourceId'] = data[i]['sourceId']
        df = df.append(row)

    df = df.reset_index()
    df.head()


def get_succinct_table():
    df = pd.DataFrame()

    # These additional columns will be kept
    columns = ['sourceId','referenceTime','elementId','value','unit','timeOffset']
    df2 = df[columns].copy()
    # Convert the time value to something Python understands
    df2['referenceTime'] = pd.to_datetime(df2['referenceTime'])

    # Preview the result
    df2.head()
 

@app.route('/')
def index():
    """
    Simple index page to provide instructions.
    """
    return """
    <h1>Frost API ETL Proxy</h1>
    <p>This Flask server acts as an ETL (Extract, Transform, Load) proxy for data from the meteorologisk Intitutt .</p>
     
    """

@app.route('/api/observations')
def get_observations():
    # Issue an HTTP GET request
    endpoint = FROST_BASE_URL + "observations/v0.jsonld"
    r = requests.get(endpoint, parameters, auth=(client_id,''))
    # Extract JSON data
    json = r.json()
    # Check if the request worked, print out any errors
    if r.status_code == 200:
        print('Data retrieved from frost.met.no!')
        data = json['data']
        return data

    else:
        print('Error! Returned status code %s' % r.status_code)
        print('Message: %s' % json['error']['message'])
        print('Reason: %s' % json['error']['reason'])
        return "ERROR"

# --- Run the Flask App ---
if __name__ == '__main__':
    # To run:
    # 1. pip install Flask requests Pillow
    # 2. Set your API key:
    #    On Linux/macOS: export OPENWEATHERMAP_API_KEY="YOUR_API_KEY_HERE"
    #    On Windows (Command Prompt): set OPENWEATHERMAP_API_KEY="YOUR_API_KEY_HERE"
    #    On Windows (PowerShell): $env:OPENWEATHERMAP_API_KEY="YOUR_API_KEY_HERE"
    # 3. python your_flask_app_name.py
    # 4. Access in browser: http://127.0.0.1:5000/
    #    Example Tile: http://127.0.0.1:5000/tile/temp_new/5/10/15
    app.run(debug=True) # debug=True enables auto-reloading and better error messages
