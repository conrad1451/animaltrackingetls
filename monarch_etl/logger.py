# CHQ: Claude AI generated file

"""
logger.py
---------
Single shared logger for the entire monarch ETL package.
Import `logger` from here in every module.
"""

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("monarch_etl")