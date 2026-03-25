# retroactive_table_log.py
import os
import logging
from datetime import date
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def register_month_in_inventory(engine, date_obj, table_name, count):
    query = text("""
    INSERT INTO data_inventory (available_date, table_name, record_count)
    VALUES (:available_date, :table_name, :record_count)
    ON CONFLICT (available_date) DO UPDATE SET 
        table_name = EXCLUDED.table_name,
        record_count = EXCLUDED.record_count,
        processed_at = CURRENT_TIMESTAMP;
    """)
    with engine.begin() as conn:
        conn.execute(query, {
            "available_date": date_obj,
            "table_name": table_name,
            "record_count": count
        })

# CHQ: Claude AI updated table name to include month and year only 
def backfill_inventory(conn_string: str, month: int, year: int):
    if not conn_string:
        logger.error("Connection string is empty!")
        return

    my_calendar = {
        1: "january",   2: "february",  3: "march",
        4: "april",     5: "may",       6: "june",
        7: "july",      8: "august",    9: "september",
        10: "october",  11: "november", 12: "december",
    }

    engine = create_engine(conn_string)
    table_name = f"{my_calendar[month]}{year}"  # e.g. "september2025"
    date_obj = date(year, month, 1)             # first of the month

    try:
        with engine.connect() as conn:
            check_query = text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = :t_name
                );
            """)
            exists = conn.execute(check_query, {"t_name": table_name}).scalar()

            if exists:
                count_query = text(f'SELECT COUNT(*) FROM "{table_name}"')
                count = conn.execute(count_query).scalar()
                register_month_in_inventory(engine, date_obj, table_name, count)
                logger.info(f"Successfully registered: {table_name} ({count} records)")
            else:
                logger.warning(f"Skipping: Table {table_name} does not exist.")

    except Exception as e:
        logger.error(f"Failed to register {table_name}: {e}")

def monarch_etl_table_backfill(year, month, conn_string):
    if conn_string:
        backfill_inventory(conn_string, month, year)
    else:
        logger.error("No connection string found in environment variables.")