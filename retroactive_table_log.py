# retroactive_table_log.py
import os
import logging
import pandas as pd
from datetime import datetime, date  # FIXED: Added date
from sqlalchemy import create_engine, text  # FIXED: Added text

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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

def backfill_inventory(conn_string: str, month: int, year: int):
    if not conn_string:
        logger.error("Connection string is empty!")
        return

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

    days_in_months ={
        1: 31,
        2: 29 if (year % 4 == 0) else 28,
        3: 31,
        4: 30,
        5: 31,
        6: 30,
        7: 31,
        8: 31,
        9: 30,
        10: 31,
        11: 30,
        12: 31,
    }

    engine = create_engine(conn_string)
    # year = 2021
    # month = 12
    
    for day in range(1, (days_in_months+1)):
        day_str = f"0{day}" if day < 10 else str(day)
        table_name = f"{my_calendar[month]}{day_str}{year}"
        date_obj = date(year, month, day) # Works now because of import
        
        try:
            with engine.connect() as conn:
                # 1. Check if table exists
                check_query = text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = :t_name
                    );
                """)
                exists = conn.execute(check_query, {"t_name": table_name}).scalar()
                
                if exists:
                    # 2. Get the row count
                    count_query = text(f'SELECT count(*) FROM "{table_name}"')
                    count = conn.execute(count_query).scalar()
                    
                    # 3. Log it
                    register_date_in_inventory(engine, date_obj, table_name, count)
                    logger.info(f"Successfully backfilled: {table_name} ({count} records)")
                else:
                    logger.warning(f"Skipping: Table {table_name} does not exist.")
                    
        except Exception as e:
            logger.error(f"Failed to backfill {table_name}: {e}")
 
def monarch_etl_table_backfill(year, month, conn_string):
    # conn_string = os.getenv('XATA_DB_MONARCH') or os.getenv('DATABASE_URL')
    # conn_string = os.getenv('XATA_DB_MONARCH')
    
    if conn_string:
        backfill_inventory(conn_string, month, year)
    else:
        logger.error("No connection string found in environment variables.")