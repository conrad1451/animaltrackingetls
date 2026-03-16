# CHQ: Claude AI generated file

"""
enrichment.py
-------------
Attaches derived data to an already-cleaned DataFrame.

Responsibilities
----------------
* Geocode each row (county + city) by calling the reverse-geocoding API.
* Extract ``time_only`` from ``eventDateParsed``.

Depends on: geocode_client, logger.
Does NOT touch GBIF extraction, cleaning, or the database.
"""

import requests
import pandas as pd

from geocode_client import fetch_location_for_coordinates, parse_county_and_city
from logger import logger


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def attach_geocoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Iterate over rows that have valid coordinates and populate the
    ``county`` and ``cityOrTown`` columns via the reverse-geocoding API.

    Rows that fail geocoding are left with ``None`` in those columns.
    """
    if "county" not in df.columns:
        df["county"] = None
    if "cityOrTown" not in df.columns:
        df["cityOrTown"] = None

    eligible = df["decimalLatitude"].notna() & df["decimalLongitude"].notna()
    if not eligible.any():
        logger.info("No rows with valid coordinates to geocode.")
        return df

    logger.info(f"Geocoding {eligible.sum()} rows…")
    for idx in df[eligible].index:
        lat = df.at[idx, "decimalLatitude"]
        lon = df.at[idx, "decimalLongitude"]
        try:
            raw = fetch_location_for_coordinates(lat, lon)
            county, city = parse_county_and_city(raw)
            df.at[idx, "county"]     = county
            df.at[idx, "cityOrTown"] = city
        except requests.exceptions.HTTPError as exc:
            logger.error(f"HTTP error geocoding ({lat},{lon}): {exc.response.status_code}")
        except requests.exceptions.RequestException as exc:
            logger.error(f"Network error geocoding ({lat},{lon}): {exc}")
        except Exception as exc:
            logger.error(f"Unexpected error geocoding ({lat},{lon}): {exc}", exc_info=True)

    return df


def attach_time_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the ``time_only`` column is present and populated from
    ``eventDateParsed``.  Safe to call even if the column already exists.
    """
    if "eventDateParsed" in df.columns and df["eventDateParsed"].notna().any():
        df["time_only"] = df["eventDateParsed"].dt.time
    else:
        df["time_only"] = None
    return df