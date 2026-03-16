# CHQ: Claude AI generated file

"""
config.py
---------
Centralizes all environment variables, API constants, and dtype mappings.
Import from here rather than scattering os.getenv() calls across modules.
"""

import os
import pandas as pd

# ---------------------------------------------------------------------------
# GBIF API
# ---------------------------------------------------------------------------
GBIF_BASE_URL = "https://api.gbif.org/v1/occurrence/search"
DEFAULT_TAXON_KEY = "5133088"  # Danaus plexippus (Monarch Butterfly)

# ---------------------------------------------------------------------------
# Neon PostgreSQL (read from environment variables)
# ---------------------------------------------------------------------------
NEON_DB_HOST = os.getenv("NEON_DB_HOST")
NEON_DB_NAME = os.getenv("NEON_DB_NAME")
NEON_DB_USER = os.getenv("NEON_DB_USER")
NEON_DB_PASSWORD = os.getenv("NEON_DB_PASSWORD")
NEON_DB_PORT = os.getenv("NEON_DB_PORT", "5432")

# ---------------------------------------------------------------------------
# Reverse-geocoding API
# ---------------------------------------------------------------------------
REVERSE_GEOCACHE_API_BASE = os.getenv("REVERSE_GEOCACHE_API_BASE")
REVERSE_GEOCACHE_API_KEY = os.getenv("REVERSE_GEOCACHE_API_KEY")

# ---------------------------------------------------------------------------
# Month name lookup (used for table naming)
# ---------------------------------------------------------------------------
MONTH_NAMES = {
    1: "january",  2: "february", 3: "march",    4: "april",
    5: "may",       6: "june",     7: "july",      8: "august",
    9: "september", 10: "october", 11: "november", 12: "december",
}

# ---------------------------------------------------------------------------
# Pandas dtype mapping (used when constructing DataFrames from raw API data)
# ---------------------------------------------------------------------------
PANDAS_DTYPE_MAP = {
    "gbifID":                        pd.StringDtype(),
    "datasetKey":                    pd.StringDtype(),
    "datasetName":                   pd.StringDtype(),
    "publishingOrgKey":              pd.StringDtype(),
    "publishingOrganizationTitle":   pd.StringDtype(),
    "eventDate":                     pd.StringDtype(),
    "eventDateParsed":               "datetime64[ns]",
    "scientificName":                pd.StringDtype(),
    "vernacularName":                pd.StringDtype(),
    "taxonKey":                      "int64",
    "kingdom":                       pd.StringDtype(),
    "phylum":                        pd.StringDtype(),
    "class":                         pd.StringDtype(),
    "order":                         pd.StringDtype(),
    "family":                        pd.StringDtype(),
    "genus":                         pd.StringDtype(),
    "species":                       pd.StringDtype(),
    "decimalLatitude":               "float64",
    "decimalLongitude":              "float64",
    "coordinateUncertaintyInMeters": "float64",
    "countryCode":                   pd.StringDtype(),
    "stateProvince":                 pd.StringDtype(),
    "locality":                      pd.StringDtype(),
    "county":                        pd.StringDtype(),
    "cityOrTown":                    pd.StringDtype(),
    "individualCount":               "int64",
    "basisOfRecord":                 pd.StringDtype(),
    "recordedBy":                    pd.StringDtype(),
    "occurrenceID":                  pd.StringDtype(),
    "collectionCode":                pd.StringDtype(),
    "catalogNumber":                 pd.StringDtype(),
    "year":                          "int64",
    "month":                         "int64",
    "day":                           "int64",
    "day_of_week":                   "int64",
    "week_of_year":                  "int64",
    "date_only":                     "object",
}

# ---------------------------------------------------------------------------
# Canonical column order (enforced before every DB write)
# ---------------------------------------------------------------------------
FINAL_COLUMNS = [
    "gbifID", "datasetKey", "publishingOrgKey", "eventDate", "eventDateParsed",
    "year", "month", "day", "day_of_week", "week_of_year", "date_only",
    "scientificName", "vernacularName", "taxonKey", "kingdom", "phylum",
    "class", "order", "family", "genus", "species", "decimalLatitude",
    "decimalLongitude", "coordinateUncertaintyInMeters", "countryCode",
    "stateProvince", "individualCount", "basisOfRecord", "recordedBy",
    "occurrenceID", "collectionCode", "catalogNumber", "county", "cityOrTown",
    "time_only",
]