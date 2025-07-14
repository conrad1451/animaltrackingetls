import os
from flask import Flask, Response, request, send_file
# import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import jsonify

# Libraries needed (pandas is not standard and must be installed in Python)
import requests
import pandas as pd

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
@app.route('/animaloccurences')
def get_observations():
    # Get parameters from the URL query string (e.g., /animaloccurences?taxonKey=5133088&country=US)
    # Provide default values if parameters are not supplied in the URL
    taxon_key = request.args.get('taxonKey', '5133088') # Default to Monarch Butterfly
    country_code = request.args.get('country', 'US')     # Default to United States

    # Define common parameters for the GBIF occurrence search
    # Using a dictionary for parameters is cleaner and handles URL encoding automatically
    params = {
        'taxonKey': taxon_key,
        'country': country_code,
        'hasCoordinate': 'true',         # Only records with coordinates
        'hasGeospatialIssue': 'false',   # Exclude records with geospatial issues
        'year': '2024',                  # Example: filter for a specific year
        'limit': '100'                   # Limit results per page
    }

    # Construct the full endpoint URL
    # requests.get will automatically append and encode the 'params' dictionary
    endpoint = GBIF_BASE_URL + "/occurrence/search"

    try:
        # Issue an HTTP GET request
        # Note: 'auth' is typically for Basic Auth. For GBIF's public search, it's not needed.
        # If you were making authenticated requests (e.g., for bulk downloads),
        # you'd use (your_gbif_username, your_gbif_password) for auth.
        r = requests.get(endpoint, params=params)

        # Extract JSON data
        json_data = r.json()

        # Check if the request worked
        if r.status_code == 200:
            print(f'Data retrieved from GBIF for taxonKey={taxon_key}, country={country_code}!')
            
            # GBIF's occurrence search results are typically in the 'results' key
            # It also includes 'offset', 'limit', 'endOfRecords', 'count'
            data = json_data.get('results', []) # Use .get() to safely access 'results'

            # You might want to return more than just the 'results' if needed,
            # or process 'results' further.
            return jsonify(data)

        else:
            # Handle API errors
            error_message = json_data.get('error', {}).get('message', 'No specific error message')
            error_reason = json_data.get('error', {}).get('reason', 'Unknown reason')
            
            print(f'Error! Returned status code {r.status_code}')
            print(f'Message: {error_message}')
            print(f'Reason: {error_reason}')
            
            # Return a JSON error response
            return jsonify({
                "error": "Failed to retrieve data from GBIF",
                "status_code": r.status_code,
                "message": error_message,
                "reason": error_reason
            }), r.status_code

    except requests.exceptions.RequestException as e:
        # Handle network or request-related errors
        print(f"Network or request error: {e}")
        return jsonify({"error": f"Network or request error: {e}"}), 500
    except json.JSONDecodeError:
        # Handle cases where the response is not valid JSON
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
