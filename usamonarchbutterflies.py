# Sources:
# [1]: https://stackoverflow.com/questions/10727366/jsonify-is-not-defined-internal-server-error

import os
from flask import Flask, Response, request, send_file, jsonify
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
 
# Libraries needed (pandas is not standard and must be installed in Python)
import requests
import pandas as pd
import json

from datetime import datetime

# CHQ: Gemini AI did the import
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# CHQ: Gemini AI added the logic for logging 
# Configure logging for better visibility of retries and other messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
 
# Define endpoint and parameters
GBIF_BASE_URL = 'https://api.gbif.org/v1'

# The 'parameters' dictionary below is for the MET Norway Frost API,
# not directly used by the GBIF API call in get_observations.
# I'll keep it here as it was in your original code, but note its context.
parameters = {
    'sources': 'SN18700,SN90450',
    'elements': 'mean(air_temperature P1D),sum(precipitation_amount P1D),mean(wind_speed P1D)',
    'referencetime': '2010-04-01/2010-04-03',
}

# These lines are for demonstration of date_conversions, not directly part of the Flask app flow
date_string_example = "2025-01-12T20:27:23"
date_object_example = datetime.strptime(date_string_example, "%Y-%m-%dT%H:%M:%S")

print(f"Original string: {date_string_example}")
print(f"Datetime object: {date_object_example}")
print(f"Type of object: {type(date_object_example)}")


# The following two functions (get_full_table, get_succinct_table)
# are not currently called or used in the Flask app.
# They also rely on a global 'data' variable which is not defined in this scope.
# If you intend to use them, they would need to take 'data' as an argument
# or be called within a function where 'data' is available.
def get_full_table(data_from_api):
    # This will return a Dataframe with all of the observations in a table format
    df = pd.DataFrame()
    for i in range(len(data_from_api)):
        # Assuming data_from_api structure similar to MET Norway data
        # For GBIF data, it would be different, often each 'entry' is an observation
        # This function might need a complete rewrite depending on data source.
        if 'observations' in data_from_api[i]: # Check if it's MET Norway format
            row = pd.DataFrame(data_from_api[i]['observations'])
            row['referenceTime'] = data_from_api[i]['referenceTime']
            row['sourceId'] = data_from_api[i]['sourceId']
            df = pd.concat([df, row], ignore_index=True) # Use pd.concat for appending
    
    # If data_from_api is already GBIF 'results' list, this function needs different logic
    # For now, assuming it's the MET Norway structure you provided earlier.
    return df.reset_index(drop=True)


def get_succinct_table(df_full):
    # Assuming df_full is the DataFrame from get_full_table
    columns = ['sourceId','referenceTime','elementId','value','unit','timeOffset']
    df2 = df_full[columns].copy()
    # Convert the time value to something Python understands
    df2['referenceTime'] = pd.to_datetime(df2['referenceTime'])
    return df2


def date_conversions(orig_date):
    # This function is fine for demonstrating date conversions
    date_string = orig_date
    date_object = datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%S")

    print(f"Original string: {date_string}")
    print(f"Datetime object: {date_object}")
    print(f"Type of object: {type(date_object)}")

    date_only = date_object.strftime("%Y-%m-%d")
    print(f"Date only (YYYY-MM-DD): {date_only}")

    time_only = date_object.strftime("%H:%M:%S")
    print(f"Time only (HH:MM:SS): {time_only}")

# CHQ: Gemini AI modified this endpoint
def filter_for_date(the_data, query_params):
    """
    Filters a list of GBIF occurrence records based on year, month, and day
    provided in query_params.

    Args:
        the_data (list): A list of dictionaries, where each dictionary is a GBIF occurrence record.
                         Expected to have an 'eventDate' key.
        query_params (dict): A dictionary containing 'year', 'month', 'day'
                             (as strings, if present).
    Returns:
        list: A new list containing only the records that match the date filters.
    """
    theOutput = []

    query_year = query_params.get('year')
    query_month = query_params.get('month')
    query_day = query_params.get('day')

    for entry in the_data:
        # GBIF occurrence records typically have 'eventDate'
        date_string_from_entry = entry.get('eventDate') 
        
        if not date_string_from_entry:
            # Skip entries without a date string
            continue

        try:
            # Parse the date string from the GBIF record
            # GBIF eventDate can sometimes be just a year or year-month,
            # so a more robust parsing might be needed for production.
            # For this example, assuming "YYYY-MM-DDTHH:MM:SS" or "YYYY-MM-DD"
            
            # Attempt to parse full datetime first
            try:
                date_object = datetime.strptime(date_string_from_entry, "%Y-%m-%dT%H:%M:%S")
            except ValueError:
                # If it's just a date, try that format
                date_object = datetime.strptime(date_string_from_entry, "%Y-%m-%d")
            
            # Extract components from the date object
            year_only = date_object.strftime("%Y")
            month_only = date_object.strftime("%m")
            day_only = date_object.strftime("%d")

            # Assume a match by default
            match = True

            # Apply filters if provided
            if query_year is not None and year_only != query_year:
                match = False
            
            if query_month is not None and month_only != query_month:
                match = False
            
            if query_day is not None and day_only != query_day:
                match = False

            if match:
                theOutput.append(entry)

        except ValueError as e:
            # Handle cases where the date string from the entry is not in the expected format
            print(f"Warning: Could not parse date '{date_string_from_entry}' from entry. Error: {e}")
            continue
        except AttributeError:
            # Handle cases where date_string_from_entry might not be a string
            print(f"Warning: 'eventDate' is not a string in entry: {entry.get('eventDate')}")
            continue

    return theOutput


# Define a retry decorator for API calls
@retry(
    wait=wait_exponential(multiplier=1, min=2, max=40), # Wait 2, 4, 8, 10, 10 seconds between retries
    stop=stop_after_attempt(5),                          # Try up to 5 times
    retry=retry_if_exception_type((
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.HTTPError # Retry on HTTP errors (e.g., 5xx server errors)
    )),
    reraise=True # Re-raise the last exception if all retries fail
)
def fetch_gbif_page(endpoint, params):
    """
    Fetches a single page of data from the GBIF API with retry logic.
    """
    logger.info(f"Attempting to fetch data from: {endpoint} with params: {params}")
    response = requests.get(endpoint, params=params)
    response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
    return response.json()


@app.route('/')
def index():
    """
    Simple index page to provide instructions.
    """
    return """
    <h1>GBIF API ETL Proxy</h1>
    <p>This Flask server acts as an ETL (Extract, Transform, Load) proxy for data from the meteorologisk Intitutt .</p>
     
    """

# CHQ: Gemini AI modified this endpoint
@app.route('/api/monarchbutterflyoccurences')
def get_observations():
    # Get parameters from the URL query string
    # e.g., /animaloccurences?taxonKey=5133088&country=US&year=2020&month=7&day=15

    taxon_key = request.args.get('taxonKey', '5133088') # Default to Monarch Butterfly
    country_code = request.args.get('country', 'US')     # Default to United States
    year_param = request.args.get('year')                # Get the year parameter (can be None)
    month_param = request.args.get('month')              # Get the month parameter (can be None)
    day_param = request.args.get('day')                  # Get the day parameter (can be None)

    # Define common parameters for the GBIF occurrence search
    params = {
        'taxonKey': taxon_key,
        'country': country_code,
        'hasCoordinate': 'true',         # Only records with coordinates
        'hasGeospatialIssue': 'false',   # Exclude records with geospatial issues
        'limit': '100'                   # Limit results per page
    }

    # Conditionally add the 'year', 'month', 'day' parameters if provided in the request URL
    if year_param:
        params['year'] = year_param
    if month_param:
        params['month'] = month_param
    if day_param:
        params['day'] = day_param

    # Construct the full endpoint URL
    endpoint = GBIF_BASE_URL + "/occurrence/search"
    all_gbif_records = []
    current_offset = 0
    end_of_records = False

    query_info = f"taxonKey={taxon_key}, country={country_code}"
    if year_param: query_info += f", year={year_param}"
    if month_param: query_info += f", month={month_param}"
    if day_param: query_info += f", day={day_param}"

    logger.info(f"Starting data retrieval for {query_info}")
    try:
        while not end_of_records:
            current_params = base_params.copy() # Create a copy to modify offset
            current_params['offset'] = current_offset

            # Fetch a page with retry logic
            page_data = fetch_gbif_page(endpoint, current_params)
            
            # Append results from the current page
            all_gbif_records.extend(page_data.get('results', []))
            
            # Check for end of records
            end_of_records = page_data.get('endOfRecords', True)
            
            # Update offset for the next page
            current_offset += page_data.get('limit', 0) # Use actual limit returned by API
            
            logger.info(f"Fetched {len(page_data.get('results', []))} records. Total fetched: {len(all_gbif_records)}. End of records: {end_of_records}")

            # Optional: Add a small delay between requests to be polite to the API
            if not end_of_records:
                time.sleep(0.1) # 100 milliseconds delay

        logger.info(f'All data retrieved from GBIF for {query_info}! Total raw records: {len(all_gbif_records)}')
        
        # Now, filter the raw data using your filter_for_date function
        # Pass the entire base_params dictionary so filter_for_date can access year, month, day
        filtered_data = filter_for_date(all_gbif_records, base_params)
        logger.info(f"Filtered records: {len(filtered_data)}")

        return jsonify(filtered_data)

    except requests.exceptions.HTTPError as e:
        error_message = f"HTTP Error: {e.response.status_code} - {e.response.text}"
        logger.error(f"Failed to retrieve data after retries: {error_message}")
        return jsonify({
            "error": "Failed to retrieve data from GBIF after multiple attempts",
            "status_code": e.response.status_code,
            "message": error_message
        }), e.response.status_code
    except requests.exceptions.RequestException as e:
        logger.error(f"Network or request error after retries: {e}")
        return jsonify({"error": f"Network or request error after multiple attempts: {e}"}), 500
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON response from GBIF: {r.text if 'r' in locals() else 'No response object'}")
        return jsonify({"error": "Invalid JSON response from GBIF"}), 500
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500

# --- Run the Flask App ---
if __name__ == '__main__':
    # To run:
    # 1. pip install Flask requests pandas Pillow tenacity
    # 2. Set your API key:
    #    On Linux/macOS: export OPENWEATHERMAP_API_KEY="YOUR_API_KEY_HERE"
    #    On Windows (Command Prompt): set OPENWEATHERMAP_API_KEY="YOUR_API_KEY_HERE"
    #    On Windows (PowerShell): $env:OPENWEATHERMAP_API_KEY="YOUR_API_KEY_HERE"
    # 3. python your_flask_app_name.py
    # 4. Access in browser: http://127.0.0.1:5000/
    #    Example Tile: http://127.0.0.1:5000/tile/temp_new/5/10/15
    app.run(debug=True) # debug=True enables auto-reloading and better error messages
