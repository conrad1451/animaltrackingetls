# butterflyetl.py (Modified)

import os
from flask import Flask, request, jsonify
import pandas as pd
import sqlite3 # Import if you choose SQLite as your data store
import logging

# Configure logging for better visibility in your Flask app
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- Configuration for Flask App ---
# Path to your processed data file/database
# IMPORTANT: Ensure this matches the 'output_file' parameter you use in etl_script.py
DATA_FILE_PATH = 'monarch_sightings.csv' # If loading from CSV
# Or, if using SQLite:
# DB_PATH = 'monarch_sightings.db'
# TABLE_NAME = 'monarch_sightings'

# --- Placeholder functions (can be removed if not used elsewhere) ---
# These functions were part of your original code but are likely not needed
# in the context of serving pre-processed data.
def get_full_table(data_from_api):
    logger.debug("get_full_table called (placeholder).")
    return pd.DataFrame() # Return empty DataFrame or implement if truly needed for something else

def get_succinct_table(df_full):
    logger.debug("get_succinct_table called (placeholder).")
    return pd.DataFrame() # Return empty DataFrame or implement if truly needed for something else

def date_conversions(orig_date):
    logger.debug(f"date_conversions called for: {orig_date} (placeholder).")
    # This helper function for demonstration purposes can stay if you want
    # to reuse it for other date string manipulations in the Flask app.
    pass

# --- Flask Endpoint to Serve PROCESSED Data ---
@app.route('/')
def index():
    """
    Simple index page to provide instructions for using the API.
    """
    return """
    <h1>Monarch Butterfly Sightings API</h1>
    <p>This API serves processed Monarch butterfly sighting data.</p>
    <p>Access data via: <code>/api/monarchbutterflyoccurences?year=2024&month=7&day=14</code></p>
    <p>You can use <code>year</code>, <code>month</code>, and <code>day</code> parameters.</p>
    <p>Data is loaded from a local file/database, updated periodically by an ETL process.</p>
    """

@app.route('/api/monarchbutterflyoccurences')
def serve_processed_observations():
    """
    Serves filtered Monarch butterfly observation data from the locally stored file.
    """
    # Get filters from the URL query string
    req_year = request.args.get('year')
    req_month = request.args.get('month')
    req_day = request.args.get('day')
    
    try:
        # --- Load data from your chosen data store ---
        # Choose ONE of the options below (CSV or SQLite)

        # Option 1: Load from CSV (simpler for initial setup)
        if not os.path.exists(DATA_FILE_PATH):
            logger.error(f"Data file not found at {DATA_FILE_PATH}. ETL process might not have run or file path is incorrect.")
            return jsonify({"error": "Data not available. ETL process might not have run yet or file path is incorrect."}), 503
        
        # Using dtype={'gbifID': str} can prevent pandas from interpreting large IDs as numbers
        # and potentially losing precision. Adjust other dtypes as needed.
        # Ensure 'eventDateParsed' is parsed as datetime objects.
        df = pd.read_csv(DATA_FILE_PATH, dtype={'gbifID': str}, parse_dates=['eventDateParsed'], infer_datetime_format=True)
        
        # Option 2: Load from SQLite (uncomment and use if you chose SQLite in etl_script.py)
        # if not os.path.exists(DB_PATH):
        #     logger.error(f"Database file not found at {DB_PATH}. ETL process might not have run.")
        #     return jsonify({"error": "Database not available. ETL process might not have run yet."}), 503
        # conn = sqlite3.connect(DB_PATH)
        # # Make sure 'monarch_sightings' matches the table name used in load_data in etl_script.py
        # df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn, parse_dates=['eventDateParsed'])
        # conn.close()
        
        logger.info(f"Loaded {len(df)} records from {DATA_FILE_PATH}.")

        # --- Apply filters from request query parameters to the loaded DataFrame ---
        filtered_df = df.copy() # Work on a copy to avoid potential SettingWithCopyWarning warnings

        # Apply year filter
        if req_year:
            try:
                # Ensure the 'year' column exists and is numeric before comparing
                if 'year' in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df['year']):
                    filtered_df = filtered_df[filtered_df['year'] == int(req_year)]
                else:
                    logger.warning(f"Year filter requested ({req_year}), but 'year' column is missing or not numeric.")
            except ValueError:
                logger.warning(f"Invalid year parameter: {req_year}")

        # Apply month filter
        if req_month:
            try:
                if 'month' in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df['month']):
                    filtered_df = filtered_df[filtered_df['month'] == int(req_month)]
                else:
                    logger.warning(f"Month filter requested ({req_month}), but 'month' column is missing or not numeric.")
            except ValueError:
                logger.warning(f"Invalid month parameter: {req_month}")

        # Apply day filter
        if req_day:
            try:
                if 'day' in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df['day']):
                    filtered_df = filtered_df[filtered_df['day'] == int(req_day)]
                else:
                    logger.warning(f"Day filter requested ({req_day}), but 'day' column is missing or not numeric.")
            except ValueError:
                logger.warning(f"Invalid day parameter: {req_day}")

        # Convert the filtered DataFrame to a list of dictionaries for JSON response
        # Using 'records' orient converts each row to a dictionary
        results = filtered_df.to_dict(orient='records')
        logger.info(f"Returning {len(results)} filtered records.")
        return jsonify(results)

    except FileNotFoundError:
        logger.error(f"Data file/database not found at specified path. Please ensure ETL ran correctly. Path: {DATA_FILE_PATH}")
        return jsonify({"error": "Data file not found. Please ensure the ETL process has run successfully."}), 500
    except pd.errors.EmptyDataError:
        logger.warning(f"Data file at {DATA_FILE_PATH} is empty. No records to display.")
        return jsonify({"message": "Data file is empty or contains no valid data. No records to display."}), 200
    except Exception as e:
        logger.error(f"An unexpected error occurred while serving data: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error: {e}"}), 500

# --- Run the Flask App ---
if __name__ == '__main__':
    # To run:
    # 1. Ensure you have the 'monarch_sightings.csv' file (or .db) created by etl_script.py in the same directory.
    # 2. pip install Flask pandas
    # 3. python app.py
    # 4. Access in browser: http://127.0.0.1:5000/api/monarchbutterflyoccurences?year=2024&month=7&day=14
    app.run(debug=True) # debug=True enables auto-reloading and better error messages