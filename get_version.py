# get_version.py

import os
import time
import logging
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
# import calendar
import math
import psycopg2

AVIEN_DB_CONNECTION = os.getenv('AVIEN_DB_CONNECTION')

def main():
    conn_string = (
        f"{AVIEN_DB_CONNECTION}"
    ) 
    conn = psycopg2.connect(conn_string)

    query_sql = 'SELECT VERSION()'

    cur = conn.cursor()
    cur.execute(query_sql)

    version = cur.fetchone()[0]
    print(version)


if __name__ == "__main__":
    main()