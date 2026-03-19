# CHQ: Claude AI generated file

"""
etl.py
------
High-level ETL orchestration.  This is the only module that should be
imported by external callers (scripts, GitHub Actions, schedulers, etc.).

Public functions
----------------
monarch_etl                  – load a whole month
monarch_etl_day_scan         – load a single day
monarch_etl_multi_day_scan   – load a range of days within one month

All three follow the same Extract → Transform → Load → Register pattern.
"""

from datetime import datetime

from .db_loader import load_dataframe
from .gbif_client import extract_gbif_data
from .inventory import register_date_via_dataframe
from .logger import logger
from .table_naming import table_name_for_day, table_name_for_month
from .transform import transform_gbif_data

from .cleaning import get_rejections, clear_rejections  # add to imports

# ---------------------------------------------------------------------------
# Whole-month ETL
# ---------------------------------------------------------------------------

def monarch_etl(year: int, month: int, conn_string: str) -> None:
    """
    Run the full ETL pipeline for every observation in *month*/*year*.

    Parameters
    ----------
    year : int
    month : int  (1–12)
    conn_string : str
        SQLAlchemy PostgreSQL connection string.
    """
    logger.info(f"=== monarch_etl START: {year}-{month:02d} (whole month) ===")

    clear_rejections()  # reset before each run

    raw_data = extract_gbif_data(
        target_year=year,
        target_month=month,
        whole_month=True,
        limiting_page_count=True,
        num_pages_to_extract=10,
    )

    if not raw_data:
        logger.info("No raw data extracted. ETL aborted.")
        return

    df = transform_gbif_data(raw_data)

    # CHQ: Claude AI: Inspect what was rejected during cleaning
    df_bad = get_rejections()
    if not df_bad.empty:
        logger.warning(f"Rejected rows: {len(df_bad)}")
        logger.warning(f"Reasons:\n{df_bad['_failure_reason'].value_counts().to_string()}")
        df_bad.to_csv(f"rejected_{year}_{month:02d}.csv", index=False)

    if df.empty:
        logger.info("Transformed DataFrame is empty. Nothing to load.")
        return

    tbl = table_name_for_month(year, month)
    load_dataframe(df, conn_string, tbl)

    logger.info(f"=== monarch_etl END: {year}-{month:02d} ===")


# ---------------------------------------------------------------------------
# Single-day ETL
# ---------------------------------------------------------------------------

def monarch_etl_day_scan(year: int, month: int, day: int, conn_string: str) -> None:
    """
    Run the full ETL pipeline for a single calendar day.

    Parameters
    ----------
    year : int
    month : int  (1–12)
    day : int    (1–31)
    conn_string : str
        SQLAlchemy PostgreSQL connection string.
    """
    logger.info(f"=== monarch_etl_day_scan START: {year}-{month:02d}-{day:02d} ===")

    clear_rejections()  # reset before each run

    raw_data = extract_gbif_data(
        target_year=year,
        target_month=month,
        whole_month=False,
        target_day=day,
        limiting_page_count=True,
        num_pages_to_extract=10,
        records_limitation=None,
    )

    if not raw_data:
        logger.info("No raw data extracted. ETL aborted.")
        return

    df = transform_gbif_data(raw_data)

    # CHQ: Claude AI: Inspect what was rejected during cleaning
    df_bad = get_rejections()
    if not df_bad.empty:
        logger.warning(f"Rejected rows: {len(df_bad)}")
        logger.warning(f"Reasons:\n{df_bad['_failure_reason'].value_counts().to_string()}")
        df_bad.to_csv(f"rejected_{year}_{month:02d}_{day:02d}.csv", index=False)

    if df.empty:
        logger.info("Transformed DataFrame is empty. Nothing to load.")
        return

    tbl = table_name_for_day(year, month, day)
    load_dataframe(df, conn_string, tbl)

    date_obj = datetime(year, month, day).date()
    register_date_via_dataframe(conn_string, date_obj, tbl, len(df))

    logger.info(f"=== monarch_etl_day_scan END: {year}-{month:02d}-{day:02d} ===")


# ---------------------------------------------------------------------------
# Multi-day ETL
# ---------------------------------------------------------------------------

def monarch_etl_multi_day_scan(
    year: int,
    month: int,
    day_start: int,
    day_end: int,
    conn_string: str,
) -> None:
    """
    Run ``monarch_etl_day_scan`` for each day in [*day_start*, *day_end*]
    (both endpoints inclusive).

    Parameters
    ----------
    year : int
    month : int      (1–12)
    day_start : int  first day to process
    day_end : int    last day to process (inclusive)
    conn_string : str
        SQLAlchemy PostgreSQL connection string.
    """
    logger.info(
        f"=== monarch_etl_multi_day_scan START: "
        f"{year}-{month:02d} days {day_start}–{day_end} ==="
    )
    for day in range(day_start, day_end + 1):
        monarch_etl_day_scan(year, month, day, conn_string)
    logger.info(f"=== monarch_etl_multi_day_scan END ===")