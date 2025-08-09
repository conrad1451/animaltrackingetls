import monarch_butterfly_module

from datetime import datetime, timedelta


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

    # This will attempt to run for the next month
    # monarch_etl(target_year, target_month)

    # You could also set specific dates for testing:
    # monarch_etl(2025, 6) # For June 2025

    # monarch_etl(2025, 5) # For May 2025
    # monarch_etl(2024, 9) # For Sep 2024
    # monarch_etl_day_scan(2025, 6, 22) # For Jun 30 2025 # had 164 entries
    # monarch_etl_day_scan(2025, 6, 26) # For Jun 26 2025 

    # monarch_etl_multi_day_scan(2025, 6, 23, 25)
    monarch_butterfly_module.monarch_etl_multi_day_scan(2025, 6, 27, 30)