# Sources:
# [1]: https://stackoverflow.com/questions/10727366/jsonify-is-not-defined-internal-server-error

import os
from flask import Flask, Response, request, send_file, jsonify # [1]
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
  
# Libraries needed (pandas is not standard and must be installed in Python)
import requests
import pandas as pd
import json

# Initialize Flask app
app = Flask(__name__)
 

# Define endpoint and parameters
GBIF_BASE_URL = 'https://api.gbif.org/v1'

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
    <h1>GBIF API ETL Proxy</h1>
    <p>This Flask server acts as an ETL (Extract, Transform, Load) proxy for data from the meteorologisk Intitutt .</p>
     
    """

# CHQ: Gemini AI generated this endpoint
@app.route('/api/monarchbutterflyoccurences')
def get_observations():
    # Get parameters from the URL query string
    # e.g., /animaloccurences?taxonKey=5133088&country=US&year=2020

    taxon_key = request.args.get('taxonKey', '5133088') # Default to Monarch Butterfly
    country_code = request.args.get('country', 'US')     # Default to United States
    year_param = request.args.get('year')                # Get the year parameter (can be None)
    
    # GBIF API allows a range of years, so let's handle that.
    # If the user provides "2020,2022", it will be passed directly.
    # If a single year like "2020" is passed, it works fine too.
    # If no year is provided, we can either default to a specific year/range or omit it
    # to get all available years. For this example, let's omit it if not provided.

    # Define common parameters for the GBIF occurrence search
    params = {
        'taxonKey': taxon_key,
        'country': country_code,
        'hasCoordinate': 'true',         # Only records with coordinates
        'hasGeospatialIssue': 'false',   # Exclude records with geospatial issues
        'limit': '100'                   # Limit results per page
    }

    # Conditionally add the 'year' parameter if it was provided in the request URL
    if year_param:
        params['year'] = year_param

    # Construct the full endpoint URL
    endpoint = GBIF_BASE_URL + "/occurrence/search"

    try:
        r = requests.get(endpoint, params=params)
        json_data = r.json()

        if r.status_code == 200:
            print(f'Data retrieved from GBIF for taxonKey={taxon_key}, country={country_code}'
                  f'{f", year={year_param}" if year_param else ""}!')
            
            data = json_data.get('results', [])
            return jsonify(data)

        else:
            error_message = json_data.get('error', {}).get('message', 'No specific error message')
            error_reason = json_data.get('error', {}).get('reason', 'Unknown reason')
            
            print(f'Error! Returned status code {r.status_code}')
            print(f'Message: {error_message}')
            print(f'Reason: {error_reason}')
            
            return jsonify({
                "error": "Failed to retrieve data from GBIF",
                "status_code": r.status_code,
                "message": error_message,
                "reason": error_reason
            }), r.status_code

    except requests.exceptions.RequestException as e:
        print(f"Network or request error: {e}")
        return jsonify({"error": f"Network or request error: {e}"}), 500
    except json.JSONDecodeError:
        print(f"Error decoding JSON response: {r.text}")
        return jsonify({"error": "Invalid JSON response from GBIF"}), 500


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
