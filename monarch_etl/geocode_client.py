# CHQ: Claude AI generated file

"""
geocode_client.py
-----------------
Client for the reverse-geocoding (part-of) API.

Responsibilities
----------------
* Build the endpoint URL from environment config.
* Fetch county / city data for a single lat/lon pair with retry logic.

Does NOT touch pandas, SQLAlchemy, or GBIF.
"""

import requests

from config import REVERSE_GEOCACHE_API_BASE, REVERSE_GEOCACHE_API_KEY
from logger import logger
from retry_config import http_retry


# ---------------------------------------------------------------------------
# Single-coordinate lookup (retryable)
# ---------------------------------------------------------------------------

@http_retry
def fetch_location_for_coordinates(latitude: float, longitude: float) -> dict:
    """
    Query the reverse-geocoding API for the county and city at *latitude*,
    *longitude*.

    Parameters
    ----------
    latitude : float
        Decimal latitude of the observation.
    longitude : float
        Decimal longitude of the observation.

    Returns
    -------
    dict
        Raw GeoJSON-style response from the API, e.g.::

            {
              "features": [
                {
                  "properties": {
                    "county": "Los Angeles",
                    "city": "Los Angeles"
                  }
                }
              ]
            }

    Raises
    ------
    ValueError
        If ``REVERSE_GEOCACHE_API_BASE`` is not set.
    requests.exceptions.HTTPError
        On non-2xx responses (after retries are exhausted).
    """
    if not REVERSE_GEOCACHE_API_BASE:
        raise ValueError("REVERSE_GEOCACHE_API_BASE environment variable is not set.")

    endpoint = (
        f"{REVERSE_GEOCACHE_API_BASE}/part-of"
        f"?lon={longitude}&lat={latitude}"
        f"&geometry=geometry_1000"
        f"&apiKey={REVERSE_GEOCACHE_API_KEY}"
    )

    logger.info(f"Reverse-geocoding lat={latitude}, lon={longitude}")
    response = requests.get(endpoint, timeout=60)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Convenience: extract county + city from a raw API response
# ---------------------------------------------------------------------------

def parse_county_and_city(raw_response: dict) -> tuple[str | None, str | None]:
    """
    Extract (county, city) strings from a ``fetch_location_for_coordinates``
    response.

    Returns (None, None) if the response contains no features.
    """
    features = raw_response.get("features", [])
    if not features:
        logger.warning("Geocoding response contained no features.")
        return None, None

    props = features[0].get("properties", {})
    return props.get("county"), props.get("city")