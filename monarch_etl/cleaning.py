# CHQ: Claude AI generated file

"""
cleaning.py
-----------
Cleans a raw GBIF DataFrame.

Responsibilities
----------------
* Parse and validate event dates — with a rescue pass for partial formats
  before dropping anything.
* Capture every dropped row into a ``_rejections`` sidecar list so callers
  can inspect, export, or re-process failed records.
* Coerce numeric coordinate and count columns.
* Drop records that truly cannot be salvaged.

Does NOT enrich with geocoding or touch the database.

Rejection sidecar
-----------------
Every step that drops rows appends to a module-level list ``_rejections``.
After running ``clean_raw_dataframe`` you can retrieve it with
``get_rejections()`` and clear it with ``clear_rejections()``.

Example
-------
    from monarch_etl.cleaning import clean_raw_dataframe, get_rejections, clear_rejections

    clear_rejections()                        # reset from any prior run
    df_clean = clean_raw_dataframe(df_raw)
    df_bad   = get_rejections()               # DataFrame of all dropped rows

    df_bad.to_csv("rejected_rows.csv", index=False)   # inspect or archive

    # Break down why each row was rejected
    print(df_bad["_failure_reason"].value_counts())
    print(df_bad["_failure_detail"].value_counts())
"""

import pandas as pd

from .logger import logger


# ---------------------------------------------------------------------------
# Rejection sidecar  (module-level accumulator)
# ---------------------------------------------------------------------------

_rejections: list[dict] = []

# CHQ: Claude AI generated to recapture and log failed rows
def get_rejections() -> pd.DataFrame:
    """Return all rows dropped during the most recent cleaning run."""
    return pd.DataFrame(_rejections) if _rejections else pd.DataFrame()

# CHQ: Claude AI generated to clear failed rows
def clear_rejections() -> None:
    """Reset the rejection sidecar (call before each ETL run)."""
    _rejections.clear()

# CHQ: Claude AI generated to reject failed rows
def _reject(rows: pd.DataFrame, reason: str, detail: str = "") -> None:
    """Append *rows* to the sidecar with a reason tag."""
    if rows.empty:
        return
    tagged = rows.copy()
    tagged["_failure_reason"] = reason
    tagged["_failure_detail"] = detail
    _rejections.extend(tagged.to_dict(orient="records"))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_raw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps to *df* and return the cleaned result.

    Steps
    -----
    1. Rescue partial / non-standard ``eventDate`` formats before parsing.
    2. Parse ``eventDate`` → ``eventDateParsed`` (UTC-aware datetime).
    3. Reject + record rows where the date still cannot be parsed.
    4. Coerce ``decimalLatitude`` / ``decimalLongitude`` to float; reject NaN.
    5. Coerce ``individualCount`` to int (default 1 on failure).
    6. Derive temporal sub-columns from ``eventDateParsed``.
    7. Select and return only the canonical final column set.

    Dropped rows are accessible via ``get_rejections()`` after this call.
    """
    df = _rescue_event_dates(df)
    df = _parse_event_dates(df)
    df = _clean_coordinates(df)
    df = _clean_individual_count(df)
    df = _derive_temporal_columns(df)
    df = _select_final_columns(df)
    return df


# ---------------------------------------------------------------------------
# Private step functions
# ---------------------------------------------------------------------------
# CHQ: Claude AI generated to rescue rows that failed due to event date
def _rescue_event_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise known non-standard ``eventDate`` formats *in place* before the
    strict ``pd.to_datetime`` parse runs.

    Formats handled
    ---------------
    * ``"YYYY"``            → ``"YYYY-01-01"``  (year-only; assume Jan 1)
    * ``"YYYY-MM"``         → ``"YYYY-MM-01"``  (year+month; assume day 1)
    * ``"YYYY-MM-DD/..."``  → ``"YYYY-MM-DD"``  (date range; take start date)
    * ``"YYYY/..."``        → ``"YYYY-01-01"``  (year range; take start year)

    Rows that are ``None`` / ``NaN`` / empty are left unchanged so that
    ``_parse_event_dates`` can tag and reject them properly.
    """
    if "eventDate" not in df.columns:
        return df

    raw = df["eventDate"].astype(str).str.strip()
 
    def _fix(val: str) -> str:
        """Return a normalised ISO date string, or the original value."""
        if not val or val.lower() in ("none", "nan", "nat", ""):
            return val

        # Date range  →  take start date
        if "/" in val:
            val = val.split("/")[0].strip()

        # Year-only  →  assume Jan 1
        if len(val) == 4 and val.isdigit():
            return f"{val}-01-01"

        # Year-month  →  assume day 1
        if len(val) == 7 and val[4] == "-":
            return f"{val}-01"

        return val

    fixed = raw.apply(_fix)

    # CHQ: Claude AI: debug — remove after diagnosis
    changed_mask = fixed != raw
    if changed_mask.any():
        sample = pd.DataFrame({
            "original": raw[changed_mask],
            "fixed":    fixed[changed_mask],
        }).head(20)
        logger.info(f"Sample of rescued/modified eventDate values:\n{sample.to_string()}")

    # also log a sample of values that were NOT changed, to confirm they look right
    unchanged_sample = raw[~changed_mask].head(10).tolist()
    logger.info(f"Sample of unchanged eventDate values (will go to parser as-is): {unchanged_sample}")
    
    rescued_count = changed_mask.sum()

    if rescued_count:
        df = df.copy()
        df.loc[changed_mask, "eventDate"] = fixed[changed_mask]
        logger.info(
            f"Rescued {rescued_count} rows with non-standard eventDate formats "
            f"(year-only, year-month, or date-range)."
        )

    return df


def _parse_event_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["eventDateParsed"] = pd.to_datetime(df["eventDate"], errors="coerce")
    
    # Localize naive datetimes to UTC, or convert tz-aware ones to UTC
    def _fix_timezone(dt):
        if dt is pd.NaT:
            return dt
        if dt.tzinfo is None:
            return dt.tz_localize("UTC")
        return dt.tz_convert("UTC")

    df["eventDateParsed"] = df["eventDateParsed"].apply(_fix_timezone)

    bad_mask = df["eventDateParsed"].isna()
    if bad_mask.any():
        bad_rows = df[bad_mask].copy()
        bad_rows["_raw_eventDate"] = bad_rows["eventDate"]   # preserve original
        _reject(
            bad_rows,
            reason="unparseable_eventDate",
            detail="pd.to_datetime could not parse eventDate after rescue pass",
        )
        logger.warning(
            f"Dropped {bad_mask.sum()} rows with unparseable eventDate. "
            f"Unique raw values: {df.loc[bad_mask, 'eventDate'].unique()[:10]}"
        )

    df = df[~bad_mask].copy()
    logger.info(f"After date parsing: {len(df)} records.")
    return df


def _clean_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["decimalLatitude"]  = pd.to_numeric(df["decimalLatitude"],  errors="coerce")
    df["decimalLongitude"] = pd.to_numeric(df["decimalLongitude"], errors="coerce")

    bad_mask = df["decimalLatitude"].isna() | df["decimalLongitude"].isna()
    if bad_mask.any():
        _reject(
            df[bad_mask],
            reason="invalid_coordinates",
            detail="decimalLatitude or decimalLongitude is null / non-numeric",
        )
        logger.warning(f"Dropped {bad_mask.sum()} rows with invalid coordinates.")

    return df[~bad_mask].copy()


def _clean_individual_count(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
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
    df = df.copy()
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