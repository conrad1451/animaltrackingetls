# CHQ: Claude AI generated file

"""
inventory.py
------------
Manages the ``data_inventory`` table that tracks which dates have been
processed and how many records each run produced.

Responsibilities
----------------
* Upsert a single date record (insert or update on conflict).
* Delete + re-insert pattern as an alternative upsert strategy.

Depends on: sqlalchemy, logger.
"""

import pandas as pd
from sqlalchemy import create_engine, text
from datetime import date as date_type

from .logger import logger


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_date(
    conn_string: str,
    date_obj: date_type,
    table_name: str,
    record_count: int,
) -> None:
    """
    Record (or update) a processed date in ``data_inventory``.

    Uses an ``ON CONFLICT … DO UPDATE`` upsert so re-running a day's ETL
    simply refreshes the row rather than raising a duplicate-key error.

    Parameters
    ----------
    conn_string : str
        SQLAlchemy-compatible PostgreSQL connection string.
    date_obj : date
        The observation date that was processed.
    table_name : str
        Name of the table where the day's records were written.
    record_count : int
        Number of rows successfully loaded.
    """
    engine = create_engine(conn_string)
    query = text("""
        INSERT INTO data_inventory (available_date, table_name, record_count)
        VALUES (:available_date, :table_name, :record_count)
        ON CONFLICT (available_date) DO UPDATE SET
            table_name   = EXCLUDED.table_name,
            record_count = EXCLUDED.record_count,
            processed_at = CURRENT_TIMESTAMP;
    """)
    with engine.begin() as conn:
        conn.execute(query, {
            "available_date": date_obj,
            "table_name":     table_name,
            "record_count":   int(record_count),
        })
    logger.info(f"Inventory updated: {date_obj} → '{table_name}' ({record_count} rows).")


def register_date_via_dataframe(
    conn_string: str,
    date_obj: date_type,
    table_name: str,
    record_count: int,
) -> None:
    """
    Alternative upsert using a pandas DELETE + INSERT pattern.

    Prefer ``register_date`` for new code; this variant is kept for
    backward compatibility with callers that relied on the old behaviour.
    """
    engine = create_engine(conn_string)
    clean_date = pd.to_datetime(date_obj).date()

    delete_query = text("DELETE FROM data_inventory WHERE available_date = :date_val")
    with engine.connect() as conn:
        conn.execute(delete_query, {"date_val": clean_date})
        conn.commit()
        logger.info(f"Cleared existing inventory record for {clean_date}.")

    inventory_df = pd.DataFrame({
        "available_date": [clean_date],
        "table_name":     [table_name],
        "record_count":   [int(record_count)],
        "processed_at":   [pd.Timestamp.now()],
    })
    inventory_df.to_sql("data_inventory", engine, if_exists="append", index=False)
    logger.info(f"Inventory re-inserted: {clean_date} → '{table_name}' ({record_count} rows).")