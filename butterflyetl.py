# butterflyetl.py (Modified for Neon PostgreSQL)

import os
from flask import Flask, request, jsonify
import pandas as pd
import logging

# For PostgreSQL connection
from sqlalchemy import create_engine
import psycopg2 # Imported by create_engine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- Neon Database Configuration (READ FROM ENVIRONMENT VARIABLES) ---
NEON_DB_HOST = os.getenv('NEON_DB_HOST')
NEON_DB_NAME = os.getenv('NEON_DB_NAME')
NEON_DB_USER = os.getenv('NEON_DB_USER')
NEON_DB_PASSWORD = os.getenv('NEON_DB_PASSWORD')
NEON_DB_PORT = os.getenv('NEON_DB_PORT', '5432') # Default to 5432 if not set

DB_TABLE_NAME = 'monarch_sightings' # Table name where data is loaded

# --- Helper function to get DB engine ---
def get_db_engine():
    if not all([NEON_DB_HOST, NEON_DB_NAME, NEON_DB_USER, NEON_DB_PASSWORD, NEON_DB_PORT]):
        logger.error("Database connection environment variables are not fully set.")
        return None
    
    db_connection_str = (
        f"postgresql+psycopg2://{NEON_DB_USER}:{NEON_DB_PASSWORD}@"
        f"{NEON_DB_HOST}:{NEON_DB_PORT}/{NEON_DB_NAME}?sslmode=require"
    )
    return create_engine(db_connection_str)

# --- Flask Endpoint to Serve PROCESSED Data from DB ---
@app.route('/')
def index():
    """
    Simple index page to provide instructions for using the API.
    """
    return """
    <h1>Monarch Butterfly Sightings API</h1>
    <p>This API serves processed Monarch butterfly sighting data from a Neon PostgreSQL database.</p>
    <p>Access data via: <code>/api/monarchbutterflyoccurences?year=2024&month=7&day=14</code></p>
    <p>You can use <code>year</code>, <code>month</code>, and <code>day</code> parameters.</p>
    <p>Data is loaded into the database periodically by an ETL process.</p>
    """

@app.route('/api/monarchbutterflyoccurences')
def serve_processed_observations():
    """
    Serves filtered Monarch butterfly observation data from the Neon PostgreSQL database.
    """
    # Get filters from the URL query string
    req_year = request.args.get('year')
    req_month = request.args.get('month')
    req_day = request.args.get('day')
    
    try:
        engine = get_db_engine()
        if engine is None:
            return jsonify({"error": "Database connection configuration missing."}), 500

        # --- Load data from the PostgreSQL database ---
        # It's more efficient to apply filters in the SQL query if possible,
        # but for simplicity and consistency with previous Pandas filtering,
        # we'll fetch all and filter in Pandas for now.
        # For very large datasets, constructing a WHERE clause for read_sql_query
        # would be significantly more efficient.
        
        # Example of fetching all (less efficient for huge tables):
        df = pd.read_sql_query(f"SELECT * FROM {DB_TABLE_NAME}", engine, parse_dates=['eventDateParsed'])
        
        logger.info(f"Loaded {len(df)} records from database table {DB_TABLE_NAME}.")

        # --- Apply filters from request query parameters to the loaded DataFrame ---
        filtered_df = df.copy()

        if req_year:
            try:
                if 'year' in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df['year']):
                    filtered_df = filtered_df[filtered_df['year'] == int(req_year)]
                else:
                    logger.warning(f"Year filter requested ({req_year}), but 'year' column is missing or not numeric.")
            except ValueError:
                logger.warning(f"Invalid year parameter: {req_year}")

        if req_month:
            try:
                if 'month' in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df['month']):
                    filtered_df = filtered_df[filtered_df['month'] == int(req_month)]
                else:
                    logger.warning(f"Month filter requested ({req_month}), but 'month' column is missing or not numeric.")
            except ValueError:
                logger.warning(f"Invalid month parameter: {req_month}")

        if req_day:
            try:
                if 'day' in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df['day']):
                    filtered_df = filtered_df[filtered_df['day'] == int(req_day)]
                else:
                    logger.warning(f"Day filter requested ({req_day}), but 'day' column is missing or not numeric.")
            except ValueError:
                logger.warning(f"Invalid day parameter: {req_day}")

        results = filtered_df.to_dict(orient='records')
        logger.info(f"Returning {len(results)} filtered records.")
        return jsonify(results)

    except Exception as e:
        logger.error(f"An unexpected error occurred while serving data from DB: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error: {e}"}), 500

if __name__ == '__main__':
    # To run:
    # 1. Ensure your Neon database environment variables are set (NEON_DB_HOST, etc.)
    # 2. pip install Flask pandas psycopg2-binary sqlalchemy
    # 3. Run etl_script.py FIRST to populate the database.
    # 4. python app.py
    # 5. Access in browser: http://127.0.0.1:5000/api/monarchbutterflyoccurences?year=2024&month=7&day=14
    app.run(debug=True)