# etl_script.py

import monarch_butterfly_module
import os

from datetime import datetime, timedelta

# --- Neon Database Configuration (READ FROM ENVIRONMENT VARIABLES) ---
# Ensure these environment variables are set in your GitHub Actions secrets or local environment
NEON_DB_HOST = os.getenv('NEON_DB_HOST')
NEON_DB_NAME = os.getenv('NEON_DB_NAME')
NEON_DB_USER = os.getenv('NEON_DB_USER')
NEON_DB_PASSWORD = os.getenv('NEON_DB_PASSWORD')
NEON_DB_PORT = os.getenv('NEON_DB_PORT', '5432')

GOOGLE_VM_DOCKER_HOSTED_SQL = os.getenv('GOOGLE_VM_DOCKER_HOSTED_SQL', '5432')
 
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

    conn_string_gcp_docker = GOOGLE_VM_DOCKER_HOSTED_SQL

    # conn_string = conn_string_neon
    conn_string = conn_string_gcp_docker
    # This will attempt to run for the next month
    # monarch_etl(target_year, target_month)


    # monarch_butterfly_module.monarch_etl_multi_day_scan(2025, 5, 1, 7, conn_string)
    # monarch_butterfly_module.monarch_etl_multi_day_scan(2025, 5, 8, 10, conn_string)
    # monarch_butterfly_module.monarch_etl_multi_day_scan(2025, 5, 11, 12, conn_string)
    monarch_butterfly_module.monarch_etl_multi_day_scan(2025, 5, 13, 15, conn_string)

    # monarch_butterfly_module.monarch_etl_multi_day_scan(2025, 6, 1, 5, conn_string)
    # monarch_butterfly_module.monarch_etl_multi_day_scan(2025, 6, 6, 10, conn_string)

    # monarch_butterfly_module.monarch_etl_multi_day_scan(2025, 6, 11, 17, conn_string)
    # monarch_butterfly_module.monarch_etl_multi_day_scan(2025, 6, 18, 20, conn_string)
    # monarch_butterfly_module.monarch_etl_multi_day_scan(2025, 6, 21, 22, conn_string)
    # monarch_etl_multi_day_scan(2025, 6, 23, 25)

    # monarch_butterfly_module.monarch_etl_multi_day_scan(2025, 6, 27, 30, conn_string)