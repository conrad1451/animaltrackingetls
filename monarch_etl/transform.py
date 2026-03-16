# CHQ: Claude AI generated file

"""
transform.py
------------
Thin orchestration layer that wires together the individual transformation
steps (cleaning → enrichment → schema enforcement) into a single callable.

Responsibilities
----------------
* Accept raw record dicts from the GBIF client.
* Return a fully transformed, schema-compliant DataFrame ready for loading.

Depends on: cleaning, enrichment, schema, logger.
"""

import pandas as pd

from .cleaning import clean_raw_dataframe
from .enrichment import attach_geocoding, attach_time_only
from .logger import logger
from .schema import enforce_schema


def transform_gbif_data(raw_data: list[dict]) -> pd.DataFrame:
    """
    Full transformation pipeline: clean → geocode → time → schema.

    Parameters
    ----------
    raw_data : list[dict]
        Raw occurrence records as returned by ``gbif_client.extract_gbif_data``.

    Returns
    -------
    pd.DataFrame
        Cleaned, enriched, schema-consistent DataFrame, or an empty
        DataFrame if *raw_data* is empty.
    """
    if not raw_data:
        logger.warning("No raw data to transform.")
        return pd.DataFrame()

    logger.info(f"Starting transformation of {len(raw_data)} records.")

    df = pd.DataFrame(raw_data)
    df = clean_raw_dataframe(df)
    df = attach_geocoding(df)
    df = attach_time_only(df)
    df = enforce_schema(df)

    logger.info(f"Transformation complete. Output rows: {len(df)}.")
    return df