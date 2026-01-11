# retroactive_table_log.py

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
from sqlalchemy import create_engine
from sqlalchemy.types import BigInteger


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# CHQ: Gemini AI created function to format table as dataframe and then convert to sql
def register_date_in_inventory_as_df(engine, date_obj, table_name, count):
    # Create a dictionary for the single row
    inventory_data = {
        'available_date': [date_obj],
        'table_name': [table_name],
        'record_count': [count],
        'processed_at': [pd.Timestamp.now()]
    }
    
    inventory_df = pd.DataFrame(inventory_data)
    
    # Load it using to_sql just like your other data
    # Note: Use if_exists='append'
    inventory_df.to_sql('data_inventory', engine, if_exists='append', index=False)
    logger.info(f"Inventory updated for {date_obj} via DataFrame.")



def backfill_december_inventory(conn_string):
    engine = create_engine(conn_string)
    year = 2021
    month = 12
    
    # Loop through all possible days in December
    for day in range(1, 32):
        day_str = f"0{day}" if day < 10 else str(day)
        table_name = f"december{day_str}{year}"
        date_obj = date(year, month, day)
        
        try:
            with engine.connect() as conn:
                # 1. Check if the table actually exists in the database
                check_query = text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = :t_name
                    );
                """)
                exists = conn.execute(check_query, {"t_name": table_name}).scalar()
                
                if exists:
                    # 2. Get the row count
                    count_query = text(f"SELECT count(*) FROM {table_name}")
                    count = conn.execute(count_query).scalar()
                    
                    # 3. Use your existing function to log it
                    register_date_in_inventory(engine, date_obj, table_name, count)
                    logger.info(f"Successfully backfilled: {table_name} ({count} records)")
                else:
                    logger.warning(f"Skipping: Table {table_name} does not exist.")
                    
        except Exception as e:
            logger.error(f"Failed to backfill {table_name}: {e}")

if __name__ == "__main__":
    # Use your existing connection string
    MY_CONN_STRING = "your_connection_string_here"
    backfill_december_inventory(MY_CONN_STRING)