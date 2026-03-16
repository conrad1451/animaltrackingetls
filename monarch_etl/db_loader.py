# CHQ: Claude AI generated file

"""
db_loader.py
------------
Writes a transformed DataFrame to the Neon PostgreSQL database.

Responsibilities
----------------
* Create a SQLAlchemy engine from a connection string.
* Write a DataFrame to a named table (replace mode).

Does NOT transform data or manage the data-inventory table.
"""

import pandas as pd
from sqlalchemy import create_engine

from .logger import logger
from .schema import SQLALCHEMY_DTYPE_MAP


def load_dataframe(df: pd.DataFrame, conn_string: str, table_name: str) -> None:
    """
    Write *df* to *table_name* in the target PostgreSQL database.

    Uses ``if_exists='replace'`` – the table is dropped and recreated on each
    call, which is appropriate for idempotent daily loads.

    Parameters
    ----------
    df : pd.DataFrame
        Fully transformed and schema-consistent DataFrame.
    conn_string : str
        SQLAlchemy-compatible connection string, e.g.
        ``"postgresql+psycopg2://user:pw@host:5432/dbname"``.
    table_name : str
        Target table name (e.g. ``"june012025"``).

    Raises
    ------
    Exception
        Any SQLAlchemy / psycopg2 error is logged and re-raised so the
        calling ETL function can decide how to handle it.
    """
    if df.empty:
        logger.info("load_dataframe: nothing to load (empty DataFrame).")
        return

    try:
        engine = create_engine(conn_string)
        logger.info(f"Loading {len(df)} rows into table '{table_name}'…")
        df.to_sql(
            table_name,
            engine,
            if_exists="replace",
            index=False,
            dtype=SQLALCHEMY_DTYPE_MAP,
        )
        logger.info(f"Successfully loaded {len(df)} rows into '{table_name}'.")
    except Exception as exc:
        logger.error(f"Error loading data into '{table_name}': {exc}", exc_info=True)
        raise