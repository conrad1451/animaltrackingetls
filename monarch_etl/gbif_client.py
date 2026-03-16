# CHQ: Claude AI generated file

"""
gbif_client.py
--------------
Low-level GBIF API client.

Responsibilities
----------------
* Fetch a single page of occurrence data with retry logic.
* Extract multiple pages up to a configurable limit.

Nothing in here touches the database or does data transformation.
"""

import time

import requests

from .config import DEFAULT_TAXON_KEY, GBIF_BASE_URL
from .logger import logger
from .retry_config import http_retry


# ---------------------------------------------------------------------------
# Single-page fetch (retryable)
# ---------------------------------------------------------------------------

@http_retry
def fetch_gbif_page(endpoint: str, params: dict) -> dict:
    """
    Fetch one page of results from the GBIF occurrence-search endpoint.

    Parameters
    ----------
    endpoint : str
        Full URL of the GBIF endpoint.
    params : dict
        Query parameters (taxonKey, country, offset, limit, …).

    Returns
    -------
    dict
        Parsed JSON response.
    """
    logger.info(f"Fetching GBIF page: {endpoint} | params={params}")
    response = requests.get(endpoint, params=params)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Multi-page extraction
# ---------------------------------------------------------------------------

def extract_gbif_data(
    taxon_key: str = DEFAULT_TAXON_KEY,
    country: str = "US",
    has_coordinate: str = "true",
    has_geospatial_issue: str = "false",
    limit_per_request: int = 300,
    target_year: int = 2025,
    target_month: int = 6,
    whole_month: bool = True,
    target_day: int = 10,
    num_pages_to_extract: int | None = None,
    limiting_page_count: bool | None = None,
    records_limitation: int | None = None,
) -> list[dict]:
    """
    Pull occurrence records from GBIF across as many pages as needed.

    Parameters
    ----------
    taxon_key : str
        GBIF taxon identifier.
    country : str
        Two-letter ISO country code.
    has_coordinate : str
        "true" / "false" – filter for records with coordinates.
    has_geospatial_issue : str
        "true" / "false" – filter records flagged with geospatial issues.
    limit_per_request : int
        Records per API call (GBIF max = 300).
    target_year : int
        Calendar year to extract.
    target_month : int
        Month (1–12) to extract.
    whole_month : bool
        If False, also filter by ``target_day``.
    target_day : int
        Day of month (only used when ``whole_month`` is False).
    num_pages_to_extract : int | None
        Hard cap on the number of pages fetched.
    limiting_page_count : bool | None
        When True, ``num_pages_to_extract`` is enforced as a hard stop.
    records_limitation : int | None
        Slice each raw page to at most this many records (useful for testing).

    Returns
    -------
    list[dict]
        Flat list of raw occurrence record dicts.
    """
    all_records: list[dict] = []
    offset = 0
    end_of_records = False
    pages_fetched = 0

    base_params = {
        "taxonKey": taxon_key,
        "country": country,
        "hasCoordinate": has_coordinate,
        "hasGeospatialIssue": has_geospatial_issue,
        "limit": limit_per_request,
        "year": target_year,
        "month": target_month,
    }
    if not whole_month:
        base_params["day"] = target_day

    while not end_of_records:
        if num_pages_to_extract is not None and pages_fetched >= num_pages_to_extract:
            logger.info(f"Page cap ({num_pages_to_extract}) reached. Stopping.")
            break

        params = {**base_params, "offset": offset}
        try:
            data = fetch_gbif_page(GBIF_BASE_URL, params)

            raw_records = data.get("results", [])
            records = _apply_record_limit(raw_records, records_limitation)

            all_records.extend(records)
            end_of_records = data.get("endOfRecords", True)
            offset += len(records)
            pages_fetched += 1

            logger.info(
                f"Page {pages_fetched}: fetched {len(records)} records "
                f"(total={len(all_records)}, endOfRecords={end_of_records})"
            )

            if limiting_page_count and num_pages_to_extract is not None and pages_fetched >= num_pages_to_extract:
                logger.info(f"limiting_page_count stop at page {pages_fetched}.")
                break

            if not end_of_records and records:
                time.sleep(0.5)  # polite delay
            elif not records and offset > 0:
                end_of_records = True

        except requests.exceptions.HTTPError as exc:
            logger.error(f"HTTP error: {exc.response.status_code} – {exc.response.text}")
            break
        except requests.exceptions.RequestException as exc:
            logger.error(f"Network error: {exc}")
            break
        except Exception as exc:
            logger.error(f"Unexpected error: {exc}", exc_info=True)
            break

    logger.info(f"Extraction complete. Total raw records: {len(all_records)}")
    return all_records


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _apply_record_limit(records: list[dict], limit: int | None) -> list[dict]:
    """Return a (possibly sliced) copy of *records*."""
    if limit is not None:
        return records[: int(limit)]
    return records