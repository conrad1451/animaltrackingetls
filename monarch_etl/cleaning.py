# CHQ: Claude AI generated file

"""
cleaning.py
-----------
Cleans a raw GBIF DataFrame.

Responsibilities
----------------
* Parse and validate event dates.
* Coerce numeric coordinate and count columns.
* Drop records that cannot be salvaged.

Does NOT enrich with geocoding or touch the database.
"""

import pandas as pd

from .logger import logger


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_raw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps to *df* and return the cleaned result.

    Steps
    -----
    1. Parse ``eventDate`` → ``eventDateParsed`` (UTC-aware datetime).
    2. Drop rows where the date cannot be parsed.
    3. Coerce ``decimalLatitude`` / ``decimalLongitude`` to float; drop NaN.
    4. Coerce ``individualCount`` to int (default 1 on failure).
    5. Derive temporal sub-columns from ``eventDateParsed``.
    6. Select and return only the canonical final column set.
    """
    df = _parse_event_dates(df)
    df = _clean_coordinates(df)
    df = _clean_individual_count(df)
    df = _derive_temporal_columns(df)
    df = _select_final_columns(df)
    return df


# ---------------------------------------------------------------------------
# Private step functions
# ---------------------------------------------------------------------------

def _parse_event_dates(df: pd.DataFrame) -> pd.DataFrame:
    df["eventDateParsed"] = pd.to_datetime(df["eventDate"], errors="coerce", utc=True)
    before = len(df)
    df.dropna(subset=["eventDateParsed"], inplace=True)
    dropped = before - len(df)
    if dropped:
        logger.warning(f"Dropped {dropped} rows with unparseable eventDate.")
    logger.info(f"After date parsing: {len(df)} records.")
    return df


def _clean_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    df["decimalLatitude"] = pd.to_numeric(df["decimalLatitude"], errors="coerce")
    df["decimalLongitude"] = pd.to_numeric(df["decimalLongitude"], errors="coerce")
    before = len(df)
    df.dropna(subset=["decimalLatitude", "decimalLongitude"], inplace=True)
    dropped = before - len(df)
    if dropped:
        logger.warning(f"Dropped {dropped} rows with invalid coordinates.")
    return df


def _clean_individual_count(df: pd.DataFrame) -> pd.DataFrame:
    if "individualCount" not in df.columns:
        df["individualCount"] = 1
    else:
        df["individualCount"] = (
            pd.to_numeric(df["individualCount"], errors="coerce")
            .fillna(1)
            .astype(int)
        )
    return df


def _derive_temporal_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df["eventDateParsed"].empty:
        return df
    dt = df["eventDateParsed"].dt
    df["year"]         = dt.year
    df["month"]        = dt.month
    df["day"]          = dt.day
    df["day_of_week"]  = dt.dayofweek
    df["week_of_year"] = dt.isocalendar().week.astype(int)
    df["date_only"]    = dt.date
    df["time_only"]    = dt.time
    return df


def _select_final_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the known columns; cast gbifID to str."""
    keep = [
        "gbifID", "datasetKey", "publishingOrgKey", "eventDate", "eventDateParsed",
        "year", "month", "day", "day_of_week", "week_of_year", "date_only",
        "scientificName", "vernacularName", "taxonKey", "kingdom", "phylum",
        "class", "order", "family", "genus", "species", "decimalLatitude",
        "decimalLongitude", "coordinateUncertaintyInMeters", "countryCode",
        "stateProvince", "individualCount", "basisOfRecord", "recordedBy",
        "occurrenceID", "collectionCode", "catalogNumber",
        # enrichment columns (may not exist yet – added later)
        "county", "cityOrTown", "time_only",
    ]
    present = [c for c in keep if c in df.columns]
    result = df[present].copy()
    if "gbifID" in result.columns:
        result["gbifID"] = result["gbifID"].astype(str)
    return result