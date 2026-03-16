# CHQ: Claude AI generated file

"""
schema.py
---------
Enforces a consistent column set and order on a DataFrame before it is
written to the database.

Responsibilities
----------------
* Add any missing canonical columns (filled with None / NULL).
* Reorder columns to the canonical sequence.
* Provide the SQLAlchemy dtype map used by ``pandas.DataFrame.to_sql``.

Depends on: config.
"""

import pandas as pd
from sqlalchemy.types import BigInteger, Date, DateTime, Float, Integer, String

from .config import FINAL_COLUMNS


# ---------------------------------------------------------------------------
# Column enforcement
# ---------------------------------------------------------------------------

def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return *df* with all canonical columns present and in canonical order.

    Missing columns are added as ``None``.
    """
    for col in FINAL_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[FINAL_COLUMNS]


# ---------------------------------------------------------------------------
# SQLAlchemy dtype map
# ---------------------------------------------------------------------------

#: Pass this to ``df.to_sql(..., dtype=SQLALCHEMY_DTYPE_MAP)`` to ensure
#: correct PostgreSQL column types on every write.
SQLALCHEMY_DTYPE_MAP: dict = {
    "gbifID":                        String,
    "datasetKey":                    String,
    "datasetName":                   String,
    "publishingOrgKey":              String,
    "publishingOrganizationTitle":   String,
    "eventDate":                     String,
    "eventDateParsed":               DateTime,
    "scientificName":                String,
    "vernacularName":                String,
    "taxonKey":                      BigInteger,
    "kingdom":                       String,
    "phylum":                        String,
    "class":                         String,
    "order":                         String,
    "family":                        String,
    "genus":                         String,
    "species":                       String,
    "decimalLatitude":               Float,
    "decimalLongitude":              Float,
    "coordinateUncertaintyInMeters": Float,
    "countryCode":                   String,
    "stateProvince":                 String,
    "locality":                      String,
    "county":                        String,
    "cityOrTown":                    String,
    "individualCount":               BigInteger,
    "basisOfRecord":                 String,
    "recordedBy":                    String,
    "occurrenceID":                  String,
    "collectionCode":                String,
    "catalogNumber":                 String,
    "year":                          Integer,
    "month":                         Integer,
    "day":                           Integer,
    "day_of_week":                   Integer,
    "week_of_year":                  BigInteger,
    "date_only":                     Date,
    "time_only":                     String,  # stored as HH:MM:SS text
}