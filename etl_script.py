# etl_script.py

import os
import time
import logging
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
# import calendar
import math

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

# --- AI Endpoint Configuration (READ FROM ENVIRONMENT VARIABLE) ---
# Ensure this environment variable is set in your GitHub Actions secrets or local environment
AI_ENDPOINT_BASE_URL = os.getenv('AI_ENDPOINT_BASE_URL')

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
        
            # CHQ: Gemini AI implemented limiting page count logic    
            # Implement limiting_page_count logic
            if limiting_page_count is not None and pages_fetched >= limiting_page_count:
                logger.info(f"Reached limiting_page_count ({limiting_page_count}). Stopping extraction.")
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

def clean_data(df):

    # 1. Robust Date Parsing for 'eventDate'
    df['eventDateParsed'] = pd.NaT
    for index, row in df.iterrows():
        date_str = row.get('eventDate')
        if date_str:
            try:
                df.at[index, 'eventDateParsed'] = parse_date(date_str)
            except (ValueError, TypeError) as e:
                logger.debug(f"Could not parse eventDate '{date_str}' at index {index}. Error: {e}")

    df.dropna(subset=['eventDateParsed'], inplace=True)
    logger.info(f"After date parsing: {len(df)} records.")

    # 2. Convert coordinates to numeric, coercing errors to NaN
    df['decimalLatitude'] = pd.to_numeric(df['decimalLatitude'], errors='coerce')
    df['decimalLongitude'] = pd.to_numeric(df['decimalLongitude'], errors='coerce')

    df.dropna(subset=['decimalLatitude', 'decimalLongitude'], inplace=True)
    logger.info(f"After dropping records without valid coordinates: {len(df)} records.")

    # 3. Handle 'individualCount': Coerce to numeric, fill NaN with 1
    # --- START FIX ---
    if 'individualCount' not in df.columns:
        logger.warning("'individualCount' column not found in GBIF data. Creating with default value 1.")
        df['individualCount'] = 1
    else:
        df['individualCount'] = pd.to_numeric(df['individualCount'], errors='coerce').fillna(1).astype(int)
    # --- END FIX ---

    # --- Enrichment / Feature Engineering ---
    if not df['eventDateParsed'].empty:
        df['year'] = df['eventDateParsed'].dt.year
        df['month'] = df['eventDateParsed'].dt.month
        df['day'] = df['eventDateParsed'].dt.day
        df['day_of_week'] = df['eventDateParsed'].dt.dayofweek
        df['week_of_year'] = df['eventDateParsed'].dt.isocalendar().week.astype(int)
        df['date_only'] = df['eventDateParsed'].dt.date
    else:
        df['year'] = pd.NA
        df['month'] = pd.NA
        df['day'] = pd.NA
        df['day_of_week'] = pd.NA
        df['week_of_year'] = pd.NA
        df['date_only'] = pd.NaT

    # Define the columns you want in your final dataset.
    final_columns = [
        'gbifID', 'datasetKey', 'datasetName', 'publishingOrgKey', 'publishingOrganizationTitle',
        'eventDate', 'eventDateParsed', 'year', 'month', 'day', 'day_of_week', 'week_of_year', 'date_only',
        'scientificName', 'vernacularName', 'taxonKey', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species',
        'decimalLatitude', 'decimalLongitude', 'coordinateUncertaintyInMeters',
        'countryCode', 'stateProvince', 'locality', # 'county' and 'cityOrTown' will be added later
        'individualCount', 'basisOfRecord', 'recordedBy', 'occurrenceID', 'collectionCode', 'catalogNumber',
    ]

    # Select only the columns that exist in the DataFrame
    df_transformed = df[[col for col in final_columns if col in df.columns]].copy()

    # Convert gbifID to string to avoid potential precision loss in large integers when loading to DB
    if 'gbifID' in df_transformed.columns:
        df_transformed['gbifID'] = df_transformed['gbifID'].astype(str)

    # --- Add 'county' and 'cityOrTown' columns using the AI endpoint (BATCHED) ---
    df_transformed['county'] = None
    df_transformed['cityOrTown'] = None

    return df_transformed

def produce_batch_coordinates(the_batch_payload, the_batch_size, the_selected_batch_payload_size):
    of_minutes = 60

    all_batch_results = []

    batch_size = the_batch_size

    # Iterate through batch_payload in chunks
    for i in range(0, the_selected_batch_payload_size, batch_size):
        current_chunk = the_batch_payload[i : i + batch_size]
        try:
            chunk_results = fetch_ai_county_city_town_analysis_batch(current_chunk)
            all_batch_results.extend(chunk_results)
            logger.info(f"Processed batch {i // batch_size + 1}. Total results collected: {len(all_batch_results)}")
            # Introduce a small delay between *chunks* of batch calls to prevent overloading
            # your AI server or hitting its concurrent request limits.
            # This is separate from any internal delays the AI server might have.
            if i + batch_size < the_selected_batch_payload_size:
                # time.sleep(0.5) # Wait 0.5 seconds between batches
                time.sleep(1.2*of_minutes) # Wait 1.2 minutes between batches


        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error during batch AI endpoint call for chunk starting at index {i}: {e.response.status_code} - {e.response.text}")
            # Log the error, but try to continue with other chunks if possible
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during batch AI endpoint call for chunk starting at index {i}: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during batch AI endpoint call for chunk starting at index {i}: {e}", exc_info=True)

    return all_batch_results

 

def run_batch_analysis(the_df, coords_to_enrich):
    df_transformed = the_df
    batch_payload = []
    for index, row in coords_to_enrich.iterrows():
        uncertainty = row['coordinateUncertaintyInMeters'] if pd.notna(row['coordinateUncertaintyInMeters']) else 0
        batch_payload.append({
            "gbifID_original_index": index, # Pass the original DataFrame index to map back
            "latitude": row['decimalLatitude'],
            "longitude": row['decimalLongitude'],
            "coordinate_uncertainty": uncertainty
        })

    # selected_batch_payload_size = max(math.ceil(len(batch_payload)/10), 1)
    selected_batch_payload_size = len(batch_payload)

    logger.info(f"Sending {selected_batch_payload_size} coordinates in a batch to AI endpoint for enrichment.")
    # BATCH_SIZE = 100 # Adjust based on your AI endpoint's capacity and Gemini's rate limits
    # BATCH_SIZE = 4 # Adjust based on your AI endpoint's capacity and Gemini's rate limits
    # BATCH_SIZE = 20 # Adjust based on your AI endpoint's capacity and Gemini's rate limits
    # BATCH_SIZE = 12 # Adjust based on your AI endpoint's capacity and Gemini's rate limits
    BATCH_SIZE = 14 # Adjust based on your AI endpoint's capacity and Gemini's rate limits

    all_batch_results = produce_batch_coordinates(batch_payload, BATCH_SIZE, selected_batch_payload_size)

    # CHQ: Gemini AI added check
    # Check if the output is a list and fail if not
    if not isinstance(all_batch_results, list):
        # Print a specific error message that the workflow can grep for
        print("::error::The output of 'produce_batch_coordinates' is not a list.")
        raise TypeError("The output of 'produce_batch_coordinates' must be a list.")

    # Map the results back to the DataFrame using the original index
    for result in all_batch_results:
        original_idx = result.get('gbifID_original_index')
        county = result.get('county')
        city = result.get('city/town')
        error = result.get('error') # Check for individual errors from the AI endpoint

        if original_idx is not None and original_idx in df_transformed.index:
            if not error:
                df_transformed.at[original_idx, 'county'] = county
                df_transformed.at[original_idx, 'cityOrTown'] = city
            else:
                logger.warning(f"Error for record at original index {original_idx} (Lat: {result.get('latitude')}, Lon: {result.get('longitude')}): {error}")
                # Optionally, store the error message in the column or leave as None
                # df_transformed.at[original_idx, 'county'] = f"Error: {error}"
                # df_transformed.at[original_idx, 'cityOrTown'] = f"Error: {error}"
        else:
            logger.warning(f"Could not find original index {original_idx} in DataFrame for result: {result}")
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


def attach_city_county_info(the_df):
    df_transformed = the_df

    df_final = df_transformed

    is_running_batch_analysis = False

    # Prepare data for batch processing: select only rows with valid coordinates
    coords_to_enrich = df_transformed[
        df_transformed['decimalLatitude'].notna() &
        df_transformed['decimalLongitude'].notna()
    ].copy() # Use .copy() to ensure you're working on a copy and avoid warnings

    if not coords_to_enrich.empty:
        # Create a list of dictionaries, each containing the necessary info for the AI endpoint
        # Important: Pass a unique identifier (like gbifID) if you have duplicate lat/lon pairs
        # so you can accurately map results back.
        if is_running_batch_analysis:
            df_final = run_batch_analysis(df_transformed, coords_to_enrich)
        else:
            df_final = run_individual_analysis(df_transformed) 

    return df_final

# --- Transformation Function ---
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

    final_df = attach_city_county_info(df_transformed)

    logger.info(f"Finished enriching location data with AI endpoint.")

    logger.info(f"Finished transformation. Transformed records: {len(final_df)}.")
    return final_df
 
# --- Load Function (Placeholder for Database Interaction) ---
def load_data(df, table_name="gbif_occurrences"):
    """
    Loads the transformed DataFrame into the Neon PostgreSQL database.
    """
    if df.empty:
        logger.info("No data to load.")
        return

    # CHQ: Gemini AI moved imports in here for optimized performance
    try:
        import psycopg2
        from sqlalchemy import create_engine

        conn_string = (
            f"postgresql+psycopg2://{NEON_DB_USER}:{NEON_DB_PASSWORD}@"
            f"{NEON_DB_HOST}:{NEON_DB_PORT}/{NEON_DB_NAME}"
        )
        engine = create_engine(conn_string)

        logger.info(f"Attempting to load {len(df)} records into '{table_name}' table...")

        # CHQ: Gemini AI added types
    
        # Define column data types for PostgreSQL to prevent issues
        # Adjust types as necessary based on your database schema
        # dtype_mapping = {
        #     'gbifID': pd.api.types.StringDtype(),
        #     'datasetKey': pd.api.types.StringDtype(),
        #     'datasetName': pd.api.types.StringDtype(),
        #     'publishingOrgKey': pd.api.types.StringDtype(),
        #     'publishingOrganizationTitle': pd.api.types.StringDtype(),
        #     'eventDate': pd.api.types.StringDtype(), # Keep original string
        #     'eventDateParsed': 'datetime64[ns]',
        #     'scientificName': pd.api.types.StringDtype(),
        #     'vernacularName': pd.api.types.StringDtype(),
        #     'taxonKey': 'int64',
        #     'kingdom': pd.api.types.StringDtype(),
        #     'phylum': pd.api.types.StringDtype(),
        #     'class': pd.api.types.StringDtype(),
        #     'order': pd.api.types.StringDtype(),
        #     'family': pd.api.types.StringDtype(),
        #     'genus': pd.api.types.StringDtype(),
        #     'species': pd.api.types.StringDtype(),
        #     'decimalLatitude': 'float64',
        #     'decimalLongitude': 'float64',
        #     'coordinateUncertaintyInMeters': 'float64',
        #     'countryCode': pd.api.types.StringDtype(),
        #     'stateProvince': pd.api.types.StringDtype(),
        #     'locality': pd.api.types.StringDtype(),
        #     'county': pd.api.types.StringDtype(), # NEW COLUMN
        #     'cityOrTown': pd.api.types.StringDtype(), # NEW COLUMN
        #     'individualCount': 'int64',
        #     'basisOfRecord': pd.api.types.StringDtype(),
        #     'recordedBy': pd.api.types.StringDtype(),
        #     'occurrenceID': pd.api.types.StringDtype(),
        #     'collectionCode': pd.api.types.StringDtype(),
        #     'catalogNumber': pd.api.types.StringDtype(),
        #     'year': 'int64',
        #     'month': 'int64',
        #     'day': 'int64',
        #     'day_of_week': 'int64',
        #     'week_of_year': 'int64',
        #     'date_only': 'object' # Store as object or convert to string if only date part is needed
        # }

        # Filter dtype_mapping to include only columns present in the DataFrame
        # and ensure pandas Dtype objects are handled for to_sql
        # to_sql often prefers string names for some types.
        # Let's simplify and rely on pandas default type conversion first,
        # then if issues, specify in the DB schema or in to_sql.
        # For 'string' dtypes, pandas automatically maps to TEXT in PG.

        # Ensure all string-like columns are explicitly cast to str to prevent issues with mixed types/NAs
        for col in df.columns:
            # Check for pandas object dtype (often used for strings or mixed types)
            # and if the column isn't numeric or datetime
            if df[col].dtype == 'object' and col not in ['eventDateParsed', 'date_only']:
                df[col] = df[col].astype(str).replace({'None': None, 'nan': None}) # Convert None/NaN str to actual None

        # Convert date_only to string if the database doesn't support date object directly
        # or if you prefer string representation
        df['date_only'] = df['date_only'].astype(str).replace({'NaT': None})


        df.to_sql(table_name, engine, if_exists='append', index=False)
        logger.info(f"Successfully loaded {len(df)} records into '{table_name}'.")

    except ImportError:
        logger.error("psycopg2 or SQLAlchemy not installed. Please install them (`pip install psycopg2-binary sqlalchemy`).")
    except Exception as e:
        logger.error(f"Error loading data into database: {e}", exc_info=True)


# --- Main ETL Orchestration Function ---
def run_monarch_etl(year, month):
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
        11: "novemeber",
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
            load_data(transformed_df, my_calendar[target_month] + " " + str(target_year))
            # load_data(transformed_df, calendar.month_name[target_month] + " " + str(target_year))
        else:
            logger.info("Transformed DataFrame is empty. No data to load.")
    else:
        logger.info("No raw data extracted. ETL process aborted.")

    logger.info("--- ETL process finished ---")

# --- Main ETL Orchestration Function ---
# --- Main ETL Orchestration Function ---
def run_monarch_etl_alt(year, month, day):
    """
    Orchestrates the ETL process for Monarch Butterfly data for a given month and year.
    """

    my_calendar = {
        1: "january", 2: "february", 3: "march", 4: "april", 5: "may", 6: "june",
        7: "july", 8: "august", 9: "september", 10: "october", 11: "novemeber", 12: "december",
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
            table_name = f"{my_calendar[month]}{day}{year}" 
            load_data(transformed_df, table_name)
        else:
            logger.info("Transformed DataFrame is empty. No data to load.")
    else:
        logger.info("No raw data extracted. ETL process aborted.")

    logger.info("--- ETL process finished ---")

if __name__ == '__main__':
    # --- Example Usage for a specific month (e.g., June 2025) ---
    # For a real cron job, you might calculate year/month dynamically
    # For testing, let's use the month following the current month
    current_date = datetime.now()
    target_year = current_date.year
    target_month = current_date.month + 1
    if target_month > 12:
        target_month = 1
        target_year += 1

    # This will attempt to run for the next month
    # run_monarch_etl(target_year, target_month)

    # You could also set specific dates for testing:
    # run_monarch_etl(2025, 6) # For June 2025

    # run_monarch_etl(2025, 5) # For May 2025
    # run_monarch_etl(2024, 9) # For Sep 2024
    run_monarch_etl_alt(2025, 6, 22) # For Jun 30 2025 # had 164 entries
    # run_monarch_etl_alt(2025, 6, 26) # For Jun 26 2025 