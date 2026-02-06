# monarch_butterfly_module.py

import os
import time
import logging
import requests
import json
import pandas as pd
from dateutil.parser import parse as parse_date
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from datetime import datetime  # <--- ADD THIS LINE

# import calendar
import math

import psycopg2
# from sqlalchemy import create_engine, BigInteger
from sqlalchemy import create_engine, text
from sqlalchemy.types import BigInteger

# from dataclasses import dataclass
# from typing import Optional

dtype_mapping = {
    'gbifID': pd.StringDtype(),
    'datasetKey': pd.StringDtype(),
    'datasetName': pd.StringDtype(),
    'publishingOrgKey': pd.StringDtype(),
    'publishingOrganizationTitle': pd.StringDtype(),
    'eventDate': pd.StringDtype(), # Keep original string
    'eventDateParsed': 'datetime64[ns]',
    'scientificName': pd.StringDtype(),
    'vernacularName': pd.StringDtype(),
    'taxonKey': 'int64',
    'kingdom': pd.StringDtype(),
    'phylum': pd.StringDtype(),
    'class': pd.StringDtype(),
    'order': pd.StringDtype(),
    'family': pd.StringDtype(),
    'genus': pd.StringDtype(),
    'species': pd.StringDtype(),
    'decimalLatitude': 'float64',
    'decimalLongitude': 'float64',
    'coordinateUncertaintyInMeters': 'float64',
    'countryCode': pd.StringDtype(),
    'stateProvince': pd.StringDtype(),
    'locality': pd.StringDtype(),
    'county': pd.StringDtype(), # NEW COLUMN
    'cityOrTown': pd.StringDtype(), # NEW COLUMN
    'individualCount': 'int64',
    'basisOfRecord': pd.StringDtype(),
    'recordedBy': pd.StringDtype(),
    'occurrenceID': pd.StringDtype(),
    'collectionCode': pd.StringDtype(),
    'catalogNumber': pd.StringDtype(),
    'year': 'int64',
    'month': 'int64',
    'day': 'int64',
    'day_of_week': 'int64',
    'week_of_year': 'int64',
    'date_only': 'object' # Store as object or convert to string if only date part is needed
}

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- GBIF API Configuration ---
GBIF_BASE_URL = "https://api.gbif.org/v1/occurrence/search"
# Default taxonKey for Monarch Butterflies (Danaus plexippus)
DEFAULT_TAXON_KEY = "5133088"

# --- Neon Database Configuration (READ FROM ENVIRONMENT VARIABLES) ---
# Ensure these environment variables are set in your GitHub Actions secrets or local environment
NEON_DB_HOST = os.getenv('NEON_DB_HOST')
NEON_DB_NAME = os.getenv('NEON_DB_NAME')
NEON_DB_USER = os.getenv('NEON_DB_USER')
NEON_DB_PASSWORD = os.getenv('NEON_DB_PASSWORD')
NEON_DB_PORT = os.getenv('NEON_DB_PORT', '5432')

GOOGLE_VM_DOCKER_HOSTED_SQL = os.getenv('GOOGLE_VM_DOCKER_HOSTED_SQL', '5432')

REVERSE_GEOCACHE_API_BASE = os.getenv('REVERSE_GEOCACHE_API_BASE')
REVERSE_GEOCACHE_API_KEY = os.getenv('REVERSE_GEOCACHE_API_KEY')
 
# --- Utility Functions ---

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.HTTPError # This includes 4xx and 5xx errors from the server
    )),
    reraise=True
)
def fetch_gbif_page_etl(endpoint, params):
    """
    Fetches a single page of data from the GBIF API with retry logic.
    """
    logger.info(f"Attempting to fetch data from: {endpoint} with params: {params}")
    response = requests.get(endpoint, params=params)
    response.raise_for_status() # Will raise HTTPError for bad responses
    return response.json()  

# --- NEW: Function to call the AI endpoint in batch mode ---
@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.HTTPError # This includes 4xx and 5xx errors from the AI server
    )),
    reraise=True # Re-raise the last exception after retries are exhausted
)
def fetch_ai_county_city_town_analysis_batch(batch_of_coordinates):
    """
    Sends a batch of coordinates to the AI endpoint for county/city analysis
    and returns the results.
    """
    if not REVERSE_GEOCACHE_API_BASE:
        raise ValueError("REVERSE_GEOCACHE_API_BASE environment variable is not set.")

    # Use the new batch endpoint
    endpoint = f"{REVERSE_GEOCACHE_API_BASE}/part-of?lon=-118.2437&lat=34.0522&geometry=geometry_1000&apiKey={REVERSE_GEOCACHE_API_KEY}"

    headers = {'Content-Type': 'application/json'}
    data = json.dumps(batch_of_coordinates) # Send the list of dicts as JSON payload

    logger.info(f"Attempting to fetch batch data from API endpoint.")
    # response = requests.get(endpoint, headers=headers, data=data, timeout=60) # Add a timeout for safety
    response = requests.get(endpoint, timeout=60) # Add a timeout for safety
    response.raise_for_status() # Raise HTTPError for bad responses (e.g., 400, 500)
    return response.json()


# returns the dictionary response from the reverse geocaching API
@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.HTTPError # This includes 4xx and 5xx errors from the AI server
    )),
    reraise=True # Re-raise the last exception after retries are exhausted
)
def fetch_county_city_town_analysis_single(latitude, longitude):
    """
    Sends a pair of coordinates to the AI endpoint for county/city analysis
    and returns the resulting data object.
    """
    if not REVERSE_GEOCACHE_API_BASE:
        raise ValueError("REVERSE_GEOCACHE_API_BASE environment variable is not set.")

    # Use the new batch endpoint
    endpoint = f"{REVERSE_GEOCACHE_API_BASE}/part-of?lon={longitude}&lat={latitude}&geometry=geometry_1000&apiKey={REVERSE_GEOCACHE_API_KEY}"
 
    logger.info(f"Attempting to fetch city and state for {longitude}&lat={latitude} coordinates.")
    # response = requests.get(endpoint, headers=headers, data=data, timeout=60) # Add a timeout for safety
    response = requests.get(endpoint, timeout=60) # Add a timeout for safety
    response.raise_for_status() # Raise HTTPError for bad responses (e.g., 400, 500)

    # thedata = response.json()
    # county_location = raw_location_data['results'][0]['properties']['county']
    # thefinal = thedata.features[0]['properties']['county']
    # thefinal = thedata.features[0]['properties'] 

    # return thefinal
    return response.json()


def final_set_of_records_to_scan(the_raw_records, records_limitation):

    if records_limitation is not None:
        # in this case, it must be an integer, so return a
        # slice of the_raw_records of size records_limitation
        # FIXME: find a way to catch any errors arising from the data type
        # of records_limitation being anything other than integer
        record_size = int(records_limitation)

        return the_raw_records[0:record_size]
    else:
        # return the entirety of the_raw_records 
        return the_raw_records


# --- Extraction Function ---
def extract_gbif_data(
    taxon_key=DEFAULT_TAXON_KEY,
    country='US',
    has_coordinate='true',
    has_geospatial_issue='false',
    limit_per_request=300, # GBIF API max limit is 300
    target_year=2025,
    target_month=6,
    whole_month = True, 
    target_day=10,
    num_pages_to_extract=None,
    limiting_page_count=None,
    records_limitation=None,
    # start_date=None,
    # end_date=None
):
    """
    Extracts occurrence data from the GBIF API, supporting date range filtering.
    """
    all_records = []
    offset = 0
    end_of_records = False
    pages_fetched = 0 # To track how many pages we've actually fetched

    params = {
        'taxonKey': taxon_key,
        'country': country,
        'hasCoordinate': has_coordinate,
        'hasGeospatialIssue': has_geospatial_issue,
        'limit': limit_per_request
    }

    params['year'] = target_year
    params['month'] = target_month

    if not whole_month:
        params['day'] = target_day
        

    # Note: GBIF API's year/month filter is for the start of the period.
    # To get data for an entire month, you specify the month.
    # If a range across months/years is needed, multiple calls would be necessary
    # or rely on post-filter by eventDateParsed in transformation.
    # For now, this assumes a single month/year extraction if dates are provided.

    # Example: If you want data for July 2025:
    # start_date = datetime(2025, 7, 1)
    # end_date = datetime(2025, 7, 31)

    # GBIF API allows 'year' and 'month' parameters.
    # If only month is provided, it needs a year too.
    # If no dates, it fetches all available for taxon/country.

    while not end_of_records:
        # CHQ: Gemini AI added logic for breaking out of the loop when num pages is specified and exceeded
        if num_pages_to_extract is not None and pages_fetched >= num_pages_to_extract:
            logger.info(f"Reached num_pages_to_extract limit ({num_pages_to_extract}). Stopping extraction.")
            break

        current_params = params.copy()
        current_params['offset'] = offset
        try:
            data = fetch_gbif_page_etl(GBIF_BASE_URL, current_params)

            raw_records = data.get('results', [])
            records = final_set_of_records_to_scan(raw_records, records_limitation)

            all_records.extend(records)

            count = data.get('count', 0)
            end_of_records = data.get('endOfRecords', True)
            offset += len(records) # Use len(records) to correctly advance offset
            pages_fetched += 1 # Increment page count

            logger.info(f"Fetched {len(records)} records. Total: {len(all_records)}. Next offset: {offset}. End of records: {end_of_records}")
        
            # CHQ: made a fix in monarch butterfly module - multiple pages should now be able to be obtained
            # CHQ: Gemini AI implemented limiting page count logic    
            # Implement limiting_page_count logic
            if limiting_page_count is not None and pages_fetched >= num_pages_to_extract:
                logger.info(f"Reached limiting_page_count ({num_pages_to_extract}). Stopping extraction.")
                break # Break the loop if the limit is reached


            # Implement a small delay between GBIF API calls to be polite and avoid rate limits
            if not end_of_records and len(records) > 0:
                 time.sleep(0.5) # Half a second delay
            elif len(records) == 0 and offset > 0: # If no records but offset is not 0, it indicates no more data
                end_of_records = True

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error during GBIF extraction: {e.response.status_code} - {e.response.text}")
            break
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during GBIF extraction: {e}")
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred during GBIF extraction: {e}")
            break

    logger.info(f"Finished extraction. Total raw records extracted: {len(all_records)}")
    return all_records

# CHQ: Gemini AI modified to correct issue where failed data parsing causes 
# time and date fields to be filled with null
def clean_data(df):
    # 1. Initialize the column as a datetime type that can handle timezones
    # This prevents the "incompatible dtype" FutureWarning and ensures dates are parsed
    df['eventDateParsed'] = pd.to_datetime(df['eventDate'], errors='coerce', utc=True)
    
    # 2. Drop records that absolutely cannot be parsed
    df.dropna(subset=['eventDateParsed'], inplace=True)
    logger.info(f"After date parsing: {len(df)} records.")

    # 3. Handle coordinates and individualCount
    df['decimalLatitude'] = pd.to_numeric(df['decimalLatitude'], errors='coerce')
    df['decimalLongitude'] = pd.to_numeric(df['decimalLongitude'], errors='coerce')
    df.dropna(subset=['decimalLatitude', 'decimalLongitude'], inplace=True)

    if 'individualCount' not in df.columns:
        df['individualCount'] = 1
    else:
        df['individualCount'] = pd.to_numeric(df['individualCount'], errors='coerce').fillna(1).astype(int)

    # 4. Derive the sub-columns from the successfully parsed dates
    if not df['eventDateParsed'].empty:
        df['year'] = df['eventDateParsed'].dt.year
        df['month'] = df['eventDateParsed'].dt.month
        df['day'] = df['eventDateParsed'].dt.day
        df['day_of_week'] = df['eventDateParsed'].dt.dayofweek
        df['week_of_year'] = df['eventDateParsed'].dt.isocalendar().week.astype(int)
        df['date_only'] = df['eventDateParsed'].dt.date
        # Crucial for your frontend display
        df['time_only'] = df['eventDateParsed'].dt.time
    
    # Define columns to keep before schema enforcement
    final_columns = [
        'gbifID', 'datasetKey', 'publishingOrgKey', 'eventDate', 'eventDateParsed', 
        'year', 'month', 'day', 'day_of_week', 'week_of_year', 'date_only', 
        'scientificName', 'vernacularName', 'taxonKey', 'kingdom', 'phylum', 
        'class', 'order', 'family', 'genus', 'species', 'decimalLatitude', 
        'decimalLongitude', 'coordinateUncertaintyInMeters', 'countryCode', 
        'stateProvince', 'individualCount', 'basisOfRecord', 'recordedBy', 
        'occurrenceID', 'collectionCode', 'catalogNumber', 'county', 'cityOrTown', 'time_only'
    ]

    # Select existing columns and cast gbifID to string
    df_transformed = df[[col for col in final_columns if col in df.columns]].copy()
    if 'gbifID' in df_transformed.columns:
        df_transformed['gbifID'] = df_transformed['gbifID'].astype(str)

    return df_transformed
 
 
# CHQ: Gemini AI generated function
def run_time_analysis(the_df, event_to_extract_time_from):
    df_transformed = the_df

    # Create a new column 'time_only' to store the extracted time
    # .dt.time extracts only the time part (HH:MM:SS) from the datetime object
    df_transformed['time_only'] = event_to_extract_time_from['eventDateParsed'].dt.time
    
    return df_transformed
 

def run_individual_analysis(thedf):
    df_transformed = thedf

    # CHQ: Gemini AI debugged to allow iteration over dataframe
    for index, row in df_transformed.iterrows():        
        the_lat = row['decimalLatitude']
        the_lon = row['decimalLongitude']

        try:
            loc_data_dict = fetch_county_city_town_analysis_single(the_lat, the_lon)

            # CHQ - logic within try clause written by Gemini AI
            # First, get the 'features' list
            features = loc_data_dict.get('features', [])

            # Check if the list is not empty
            if features:
                # Get the first feature
                first_feature = features[0]
                
                # Get the 'properties' dictionary
                properties = first_feature.get('properties', {})
                
                # Extract the county and city from the properties
                county_location = properties.get('county')
                city_location = properties.get('city')

                # Update the DataFrame
                df_transformed.at[index, 'county'] = county_location
                df_transformed.at[index, 'cityOrTown'] = city_location
            else:
                # Handle the case where no features were returned
                logger.warning("API response did not contain any features for the given coordinates.")
                county_location = None
                city_location = None
 

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error during batch AI endpoint call : {e.response.status_code} - {e.response.text}")
            # Log the error, but try to continue with other chunks if possible
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during batch AI endpoint call : {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during batch AI endpoint call  : {e}", exc_info=True)

    return df_transformed



def attach_time_discovered_info(the_df):
    df_transformed = the_df

    df_final = df_transformed

    # Prepare data for batch processing: select only rows with valid coordinates
    event_date_to_parse = df_transformed[
        df_transformed['eventDateParsed'].notna()
    ].copy() # Use .copy() to ensure you're working on a copy and avoid warnings

    if not event_date_to_parse.empty:
        # Create a list of dictionaries, each containing the necessary info for the AI endpoint
        # Important: Pass a unique identifier (like gbifID) if you have duplicate lat/lon pairs
        # so you can accurately map results back.
        df_final = run_time_analysis(df_transformed, event_date_to_parse)

    return df_final


def attach_city_county_info(the_df):
    df_transformed = the_df

    df_final = df_transformed

    # Prepare data for batch processing: select only rows with valid coordinates
    coords_to_enrich = df_transformed[
        df_transformed['decimalLatitude'].notna() &
        df_transformed['decimalLongitude'].notna()
    ].copy() # Use .copy() to ensure you're working on a copy and avoid warnings

    if not coords_to_enrich.empty:
        # Create a list of dictionaries, each containing the necessary info for the AI endpoint
        # Important: Pass a unique identifier (like gbifID) if you have duplicate lat/lon pairs
        # so you can accurately map results back.
        df_final = run_individual_analysis(df_transformed) 

    return df_final

def ensure_schema_consistency(df):
    expected_cols = [
        "gbifID", "datasetKey", "publishingOrgKey", "eventDate", "eventDateParsed", 
        "year", "month", "day", "day_of_week", "week_of_year", "date_only", 
        "scientificName", "vernacularName", "taxonKey", "kingdom", "phylum", 
        "class", "order", "family", "genus", "species", "decimalLatitude", 
        "decimalLongitude", "coordinateUncertaintyInMeters", "countryCode", 
        "stateProvince", "individualCount", "basisOfRecord", "recordedBy", 
        "occurrenceID", "collectionCode", "catalogNumber", "county", "cityOrTown", "time_only"
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = None # Adds the column as NULL if missing
    return df[expected_cols] # Ensures column order is also consistent

def transform_gbif_data(raw_data):
    """
    Transforms raw GBIF occurrence data into a cleaned and enriched Pandas DataFrame.
    """
    if not raw_data:
        logger.warning("No raw data to transform.")
        return pd.DataFrame()

    logger.info(f"Starting transformation of {len(raw_data)} records.")

    df = pd.DataFrame(raw_data)
    df_transformed = clean_data(df)
    next_to_final_df = attach_city_county_info(df_transformed)
    final_df = attach_time_discovered_info(next_to_final_df)

    # CRITICAL FIX: Call schema enforcement HERE before returning
    # This ensures the daily sightings table (e.g. december082021) has all columns
    final_df = ensure_schema_consistency(final_df)

    logger.info(f"Finished transformation. Transformed records: {len(final_df)}.")
    return final_df

# CHQ: Gemini AI modified to clear existing date entry
def register_date_in_inventory_as_df(engine, date_obj, table_name, count):
    clean_date = pd.to_datetime(date_obj).date()

    delete_query = text("DELETE FROM data_inventory WHERE available_date = :date_val")
     
    with engine.connect() as conn:
        conn.execute(delete_query, {"date_val": clean_date})
        conn.commit()
        logger.info(f"Cleared existing inventory record for {clean_date}.")

    inventory_data = {
        'available_date': [clean_date],
        'table_name': [table_name],
        'record_count': [int(count)],
        'processed_at': [pd.Timestamp.now()]
    }
    
    inventory_df = pd.DataFrame(inventory_data)
    inventory_df.to_sql('data_inventory', engine, if_exists='append', index=False)
# --- Load Function (Placeholder for Database Interaction) ---
def load_data(df, conn_string, table_name="gbif_occurrences"):
    """
    Loads the transformed DataFrame into the Neon PostgreSQL database
    using a predefined dtype map to ensure correct column types.
    """
    if df.empty:
        logger.info("No data to load.")
        return

    try:
        from sqlalchemy import create_engine
        from sqlalchemy.types import String, DateTime, Float, BigInteger, Integer, Date

        engine = create_engine(conn_string)

        logger.info(f"Attempting to load {len(df)} records into '{table_name}' table...")

        # NOTE: This line is no longer necessary as the dtype mapping will handle NaT
        # df['date_only'] = df['date_only'].astype(str).replace({'NaT': None})

        # CHQ: Corrected dtype mapping to use SQLAlchemy types
        dtype_mapping = {
            'gbifID': String,
            'datasetKey': String,
            'datasetName': String,
            'publishingOrgKey': String,
            'publishingOrganizationTitle': String,
            'eventDate': String,
            'eventDateParsed': DateTime,
            'scientificName': String,
            'vernacularName': String,
            'taxonKey': BigInteger,
            'kingdom': String,
            'phylum': String,
            'class': String,
            'order': String,
            'family': String,
            'genus': String,
            'species': String,
            'decimalLatitude': Float,
            'decimalLongitude': Float,
            'coordinateUncertaintyInMeters': Float,
            'countryCode': String,
            'stateProvince': String,
            'locality': String,
            'county': String,
            'cityOrTown': String,
            'individualCount': BigInteger,
            'basisOfRecord': String,
            'recordedBy': String,
            'occurrenceID': String,
            'collectionCode': String,
            'catalogNumber': String,
            'year': Integer,
            'month': Integer,
            'day': Integer,
            'day_of_week': Integer,
            'week_of_year': BigInteger,
            'date_only': Date  # This is the key fix
        }

        # The to_sql call now includes the dtype mapping
        df.to_sql(table_name, engine, if_exists='replace', index=False, dtype=dtype_mapping)
        logger.info(f"Successfully loaded {len(df)} records into '{table_name}'.")

    except Exception as e:
        logger.error(f"Error loading data into database: {e}", exc_info=True)

# CHQ: Gemini AI fixed function to pass parameters as a dictionary 
# Inside your load logic after the data is successfully saved to the new table
def register_date_in_inventory(engine, date_obj, table_name, count):
    # 1. Use named placeholders (:key)
    query = text("""
    INSERT INTO data_inventory (available_date, table_name, record_count)
    VALUES (:available_date, :table_name, :record_count)
    ON CONFLICT (available_date) DO UPDATE SET 
        table_name = EXCLUDED.table_name,
        record_count = EXCLUDED.record_count,
        processed_at = CURRENT_TIMESTAMP;
    """)
    
    with engine.begin() as conn:
        # 2. Pass as a DICTIONARY to satisfy SQLAlchemy 2.0
        conn.execute(query, {
            "available_date": date_obj,
            "table_name": table_name,
            "record_count": count
        })
 
# --- Main ETL Orchestration Function ---
def monarch_etl(year, month, conn_string):
    """
    Orchestrates the ETL process for Monarch Butterfly data for a given month and year.
    """


    my_calendar ={
        1: "january",
        2: "february",
        3: "march",
        4: "april",
        5: "may",
        6: "june",
        7: "july",
        8: "august",
        9: "september",
        10: "october",
        11: "november",
        12: "december",
    }


    logger.info(f"\n\nRunning ETL for {year}-{month} (entire month)\n")
    logger.info("--- ETL process started ---") 
    # start_date = datetime(year, month, 1)
    # # Calculate the last day of the month
    # if month == 12:
    #     end_date = datetime(year, 12, 31)
    # else:
    #     end_date = datetime(year, month + 1, 1) - timedelta(days=1)

    logger.info("\n\n\n--- EXTRACT STEP ---\n\n\n")

    # raw_data = extract_gbif_data(target_year=year, target_month=month, whole_month=True, limiting_page_count=True, num_pages_to_extract=10, records_limitation=42)
    raw_data = extract_gbif_data(target_year=year, target_month=month, whole_month=True, limiting_page_count=True, num_pages_to_extract=10)

    if raw_data:
        logger.info("\n\n\n--- TRANSFORM STEP ---\n\n\n")
        transformed_df = transform_gbif_data(raw_data)
        if not transformed_df.empty:
            logger.info("\n\n\n--- LOAD STEP ---\n\n\n")
            load_data(transformed_df, conn_string, my_calendar[month] + " " + str(year))
            # load_data(transformed_df, conn_string, calendar.month_name[month] + " " + str(year))
        else:
            logger.info("Transformed DataFrame is empty. No data to load.")
    else:
        logger.info("No raw data extracted. ETL process aborted.")

    logger.info("--- ETL process finished ---")

# --- Main ETL Orchestration Function ---
# --- Main ETL Orchestration Function ---
def monarch_etl_day_scan(year, month, day, conn_string):
    """
    Orchestrates the ETL process for Monarch Butterfly data for a given month and year.
    """

    my_calendar = {
        1: "january", 2: "february", 3: "march", 4: "april", 5: "may", 6: "june",
        7: "july", 8: "august", 9: "september", 10: "october", 11: "november", 12: "december",
    }

    logger.info(f"\n\nRunning ETL for {year}-{month}-{day} \n")
    logger.info("--- ETL process started ---")

    logger.info("\n\n\n--- EXTRACT STEP ---\n\n\n")

    raw_data = extract_gbif_data(
        target_year=year,
        target_month=month,
        whole_month=False,
        target_day=day,
        limiting_page_count=True,
        num_pages_to_extract=10, 
        records_limitation=None
    )

    if raw_data:
        logger.info("\n\n\n--- TRANSFORM STEP ---\n\n\n")
        transformed_df = transform_gbif_data(raw_data)
        if not transformed_df.empty:
            logger.info("\n\n\n--- LOAD STEP ---\n\n\n")
            # Corrected line: pass the variables directly to the table name string

            table_name = ""

            if(day < 10):
                table_name = f"{my_calendar[month]}0{day}{year}" 
            else:
                table_name = f"{my_calendar[month]}{day}{year}" 
 
            load_data(transformed_df, conn_string, table_name)
            
            # 2. Register the completion in the inventory table
            engine = create_engine(conn_string)
            
            # Create a date object for the inventory
            date_obj = datetime(year, month, day).date()
            record_count = len(transformed_df)
            
            logger.info(f"Registering {date_obj} in data_inventory...")
            # register_date_in_inventory(engine, date_obj, table_name, record_count)
            register_date_in_inventory_as_df(engine, date_obj, table_name, record_count)
            
        else:
            logger.info("Transformed DataFrame is empty. No data to load.")
    else:
        logger.info("No raw data extracted. ETL process aborted.")

    logger.info("--- ETL process finished ---")

def monarch_etl_multi_day_scan(year, month, day_start, day_end, conn_string):
    for chosen_day in range(day_start, day_end+1):
        monarch_etl_day_scan(year, month, chosen_day, conn_string) # For Jun 30 2025 # had 164 entries