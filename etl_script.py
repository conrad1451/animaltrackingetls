# etl_script.py

import os
from monarch_etl import monarch_etl_day_scan, monarch_etl_multi_day_scan, monarch_etl

# --- Database connection string (from environment variables) ---
AIVEN_DB_MONARCH = os.getenv('AIVEN_DB_MONARCH')

if not AIVEN_DB_MONARCH:
    print("FATAL ERROR: AIVEN_DB_MONARCH environment variable is NOT SET.")
    exit(1)

conn_string = AIVEN_DB_MONARCH

# SQLAlchemy 2.0 requires 'postgresql://' not 'postgres://'
if conn_string.startswith("postgres://"):
    conn_string = conn_string.replace("postgres://", "postgresql://", 1)

if __name__ == '__main__':

    # monarch_etl_multi_day_scan(2020, 1, 1, 31, conn_string)
    # monarch_etl_multi_day_scan(2020, 2, 1, 29, conn_string)
    # monarch_etl_multi_day_scan(2020, 3, 1, 31, conn_string)
    # monarch_etl_multi_day_scan(2020, 4, 1, 30, conn_string)
    # monarch_etl_multi_day_scan(2020, 5, 1, 31, conn_string)
    # monarch_etl_multi_day_scan(2020, 6, 1, 30, conn_string)
    # monarch_etl_multi_day_scan(2020, 7, 1, 31, conn_string)
    # monarch_etl_multi_day_scan(2020, 8, 1, 31, conn_string)
    # monarch_etl_multi_day_scan(2020, 9, 1, 30, conn_string)
    # monarch_etl_multi_day_scan(2020, 10, 1, 31, conn_string)
    # monarch_etl_multi_day_scan(2020, 11, 1, 30, conn_string)
    # monarch_etl_multi_day_scan(2020, 12, 1, 31, conn_string)

    # monarch_etl_multi_day_scan(2021, 1, 1, 31, conn_string)
    # monarch_etl_multi_day_scan(2021, 2, 1, 28, conn_string)
    # monarch_etl_multi_day_scan(2021, 3, 1, 31, conn_string)
    # monarch_etl_multi_day_scan(2021, 4, 1, 30, conn_string)
    # monarch_etl_multi_day_scan(2021, 5, 1, 31, conn_string)
    # monarch_etl_multi_day_scan(2021, 6, 1, 30, conn_string)
    # monarch_etl_multi_day_scan(2021, 7, 1, 31, conn_string)
    # monarch_etl_multi_day_scan(2021, 8, 1, 31, conn_string)
    # monarch_etl_multi_day_scan(2021, 9, 1, 30, conn_string)
    # monarch_etl_multi_day_scan(2021, 10, 1, 31, conn_string)
    # monarch_etl_multi_day_scan(2021, 11, 1, 30, conn_string)
    # monarch_etl_multi_day_scan(2021, 12, 1, 31, conn_string)

    # monarch_etl_day_scan(2021, 6, 15, conn_string)
    # monarch_etl_day_scan(2021, 6, 21, conn_string)
    # monarch_etl_multi_day_scan(2021, 6, 17, 19, conn_string)
    # monarch_etl_day_scan(2021, 6, 30, conn_string)
    # monarch_etl_day_scan(2021, 10, 25, conn_string)
    # monarch_etl_day_scan(2021, 12, 1, conn_string)
    # monarch_etl_day_scan(2021, 12, 8, conn_string)
    # monarch_etl_multi_day_scan(2021, 12, 12, 15, conn_string)
    # monarch_etl_day_scan(2021, 12, 18, conn_string)
    # monarch_etl_multi_day_scan(2021, 12, 20, 22, conn_string)
    # monarch_etl_day_scan(2021, 12, 27, conn_string)
    # monarch_etl_multi_day_scan(2021, 12, 30, 31, conn_string)

    # monarch_etl_multi_day_scan(2022, 1, 1, 31, conn_string)
    # monarch_etl_multi_day_scan(2022, 2, 1, 28, conn_string)
    # monarch_etl_multi_day_scan(2022, 3, 1, 31, conn_string)
    # monarch_etl_multi_day_scan(2022, 4, 1, 30, conn_string)
    # monarch_etl_multi_day_scan(2022, 5, 1, 31, conn_string)
    # monarch_etl_multi_day_scan(2022, 6, 1, 30, conn_string)
    # monarch_etl_multi_day_scan(2022, 7, 1, 31, conn_string)
    # monarch_etl_multi_day_scan(2022, 8, 1, 31, conn_string)
    # monarch_etl_multi_day_scan(2022, 9, 1, 30, conn_string)
    # monarch_etl_multi_day_scan(2022, 10, 1, 31, conn_string)
    # monarch_etl_multi_day_scan(2022, 11, 1, 30, conn_string)
    # monarch_etl_multi_day_scan(2022, 12, 1, 31, conn_string)

    # monarch_etl_multi_day_scan(2023, 1, 1, 31, conn_string)
    # monarch_etl_multi_day_scan(2023, 2, 1, 28, conn_string)
    # monarch_etl_multi_day_scan(2023, 3, 1, 31, conn_string)
    # monarch_etl_multi_day_scan(2023, 4, 1, 30, conn_string)
    # monarch_etl_multi_day_scan(2023, 5, 1, 31, conn_string)
    # monarch_etl_multi_day_scan(2023, 6, 1, 30, conn_string)
    # monarch_etl_multi_day_scan(2023, 7, 1, 31, conn_string)
    # monarch_etl_multi_day_scan(2023, 8, 1, 31, conn_string)
    # monarch_etl_multi_day_scan(2023, 9, 1, 30, conn_string)
    # monarch_etl_multi_day_scan(2023, 10, 1, 31, conn_string)
    # monarch_etl_multi_day_scan(2023, 11, 1, 30, conn_string)
    # monarch_etl_multi_day_scan(2023, 12, 1, 31, conn_string)

    # monarch_etl_multi_day_scan(2024, 1, 1, 31, conn_string)
    # monarch_etl_multi_day_scan(2024, 2, 1, 29, conn_string)
    # monarch_etl_multi_day_scan(2024, 3, 1, 31, conn_string)
    # monarch_etl_multi_day_scan(2024, 4, 1, 30, conn_string)
    # monarch_etl_multi_day_scan(2024, 5, 1, 31, conn_string)
    # monarch_etl_multi_day_scan(2024, 6, 1, 30, conn_string)
    # monarch_etl_multi_day_scan(2024, 7, 1, 31, conn_string)
    # monarch_etl_multi_day_scan(2024, 8, 1, 31, conn_string)
    # monarch_etl_multi_day_scan(2024, 9, 1, 30, conn_string)
    # monarch_etl_multi_day_scan(2024, 10, 1, 31, conn_string)
    # monarch_etl_multi_day_scan(2024, 11, 1, 30, conn_string)
    # monarch_etl_multi_day_scan(2024, 12, 1, 31, conn_string)

    # monarch_etl_multi_day_scan(2025, 1, 1, 31, conn_string)
    # monarch_etl_multi_day_scan(2025, 2, 1, 29, conn_string)
    # monarch_etl_multi_day_scan(2025, 3, 1, 31, conn_string)
    # monarch_etl_multi_day_scan(2025, 4, 1, 30, conn_string)
    # monarch_etl_multi_day_scan(2025, 5, 1, 31, conn_string)
    # monarch_etl_multi_day_scan(2025, 6, 1, 30, conn_string)
    # monarch_etl_multi_day_scan(2025, 7, 1, 31, conn_string)
    # monarch_etl_multi_day_scan(2025, 8, 1, 31, conn_string) 
    
    monarch_etl(2025, 9, conn_string)
    # monarch_etl(2025, 10, conn_string)
    # monarch_etl(2025, 11, conn_string)
    # monarch_etl(2025, 12, conn_string)

    # monarch_etl(2026, 1, conn_string)
    # monarch_etl(2026, 2, conn_string)