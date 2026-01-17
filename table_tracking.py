# table_tracking.py

import retroactive_table_log
import os

from datetime import datetime, timedelta

# --- Neon Database Configuration (READ FROM ENVIRONMENT VARIABLES) ---
# Ensure these environment variables are set in your GitHub Actions secrets or local environment
NEON_DB_HOST = os.getenv('NEON_DB_HOST')
NEON_DB_NAME = os.getenv('NEON_DB_NAME')
NEON_DB_USER = os.getenv('NEON_DB_USER')
NEON_DB_PASSWORD = os.getenv('NEON_DB_PASSWORD')
NEON_DB_PORT = os.getenv('NEON_DB_PORT', '5432')

# GOOGLE_VM_DOCKER_HOSTED_SQL = os.getenv('GOOGLE_VM_DOCKER_HOSTED_SQL', '5432')
# DIGITAL_OCEAN_VM_DOCKER_HOSTED_SQL = os.getenv('DIGITAL_OCEAN_VM_DOCKER_HOSTED_SQL', '5432')
# DIGITAL_OCEAN_VM_DOCKER_HOSTED_SQL = os.getenv('DIGITAL_OCEAN_VM_DOCKER_HOSTED_SQL_ALT', '5432')

# Remove the default value. If the variable isn't found, it will be None.
# DIGITAL_OCEAN_VM_DOCKER_HOSTED_SQL = os.getenv('DIGITAL_OCEAN_VM_DOCKER_HOSTED_SQL_ALT')

# IBM_DOCKER_PSQL_MONARCH = os.getenv('IBM_DOCKER_PSQL_MONARCH')
XATA_DB_MONARCH = os.getenv('XATA_DB_MONARCH')

# if not IBM_DOCKER_PSQL_MONARCH:
#     print("FATAL ERROR: IBM_DOCKER_PSQL_MONARCH environment variable is NOT SET.")
#     # Exit gracefully or raise an error
#     exit(1)

if not XATA_DB_MONARCH:
    print("FATAL ERROR: XATA_DB_MONARCH environment variable is NOT SET.")
    # Exit gracefully or raise an error
    exit(1)

conn_string_xata = XATA_DB_MONARCH
# conn_string_digital_ocean = IBM_DOCKER_PSQL_MONARCH

# if not DIGITAL_OCEAN_VM_DOCKER_HOSTED_SQL:
#     print("FATAL ERROR: DIGITAL_OCEAN_VM_DOCKER_HOSTED_SQL_ALT environment variable is NOT SET.")
#     # Exit gracefully or raise an error
#     exit(1)

# conn_string_digital_ocean = DIGITAL_OCEAN_VM_DOCKER_HOSTED_SQL
# conn_string = conn_string_digital_ocean
conn_string = conn_string_xata

# REVERSE_GEOCACHE_API_BASE = os.getenv('REVERSE_GEOCACHE_API_BASE')
# REVERSE_GEOCACHE_API_KEY = os.getenv('REVERSE_GEOCACHE_API_KEY')

if __name__ == '__main__':
    # --- Example Usage for a specific month (e.g., June 2025) ---
    # For a real cron job, you might calculate year/month dynamically
    # For testing, let's use the month following the current month
    current_date = datetime.now()
    target_year = current_date.year
    target_month = current_date.month + 1
    if target_month > 12:
        target_month = 1
        target_year += 1

    conn_string_neon = (
        f"postgresql+psycopg2://{NEON_DB_USER}:{NEON_DB_PASSWORD}@"
        f"{NEON_DB_HOST}:{NEON_DB_PORT}/{NEON_DB_NAME}"
    )

    # conn_string_gcp_docker = GOOGLE_VM_DOCKER_HOSTED_SQL

    # conn_string_digital_ocean = DIGITAL_OCEAN_VM_DOCKER_HOSTED_SQL

    # conn_string = conn_string_digital_ocean
    # conn_string = conn_string_neon
    # conn_string = conn_string_gcp_docker
    # This will attempt to run for the next month
    # monarch_etl(target_year, target_month)

    # retroactive_table_log.monarch_etl_table_backfill(2021, 12, conn_string)

    # retroactive_table_log.monarch_etl_table_backfill(2022, 1, conn_string)
    # retroactive_table_log.monarch_etl_table_backfill(2022, 2, conn_string)
    # retroactive_table_log.monarch_etl_table_backfill(2022, 3, conn_string)
    # retroactive_table_log.monarch_etl_table_backfill(2022, 4, conn_string)
    # retroactive_table_log.monarch_etl_table_backfill(2022, 5, conn_string)
    # retroactive_table_log.monarch_etl_table_backfill(2022, 6, conn_string)
    # retroactive_table_log.monarch_etl_table_backfill(2022, 7, conn_string)
    # retroactive_table_log.monarch_etl_table_backfill(2022, 8, conn_string)
    # retroactive_table_log.monarch_etl_table_backfill(2022, 9, conn_string)
    # retroactive_table_log.monarch_etl_table_backfill(2022, 10, conn_string)
    # retroactive_table_log.monarch_etl_table_backfill(2022, 11, conn_string)
    # retroactive_table_log.monarch_etl_table_backfill(2022, 12, conn_string)

    # retroactive_table_log.monarch_etl_table_backfill(2023, 1, conn_string)
    # retroactive_table_log.monarch_etl_table_backfill(2023, 2, conn_string)
    retroactive_table_log.monarch_etl_table_backfill(2023, 3, conn_string)
    # retroactive_table_log.monarch_etl_table_backfill(2023, 4, conn_string)
    # retroactive_table_log.monarch_etl_table_backfill(2023, 5, conn_string)
    # retroactive_table_log.monarch_etl_table_backfill(2023, 6, conn_string)
    # retroactive_table_log.monarch_etl_table_backfill(2023, 7, conn_string)
    # retroactive_table_log.monarch_etl_table_backfill(2023, 8, conn_string)
    # retroactive_table_log.monarch_etl_table_backfill(2023, 9, conn_string)