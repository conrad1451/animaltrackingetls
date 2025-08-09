# etl_past_day_script.py

import monarch_butterfly_module

from datetime import date, datetime, timedelta


# CHQ: Gemini AI generated function
def get_first_sunday_of_year(input_date: date) -> date:
    """
    For a given date, this function returns the date of the first Sunday
    of the year in which the input date falls.

    Args:
        input_date (date): A datetime.date object.

    Returns:
        date: A datetime.date object representing the first Sunday of the year.
    """
    # Step 1: Get the first day of the year from the input date's year
    first_day_of_year = date(input_date.year, 1, 1)

    # Step 2: Find the weekday of January 1st.
    # Monday is 1, Sunday is 7.
    jan_1_weekday = first_day_of_year.isoweekday()

    # Step 3: Calculate how many days to add to get to the first Sunday.
    # The number of days to add is 7 minus the weekday number.
    # If Jan 1st is a Sunday (7), we add 0 days.
    # If Jan 1st is a Monday (1), we add 6 days.
    # If Jan 1st is a Saturday (6), we add 1 day.
    days_to_add = (7 - jan_1_weekday) % 7

    # Step 4: Add the days to January 1st to get the first Sunday
    first_sunday = first_day_of_year + timedelta(days=days_to_add)

    return first_sunday



if __name__ == '__main__':
    
        
    # --- Example Usage ---
    # Use a date from a year to test the function
    test_date = date(2025, 8, 9)
    first_sunday_2025 = get_first_sunday_of_year(test_date)
    print(f"The first Sunday of the year {test_date.year} is: {first_sunday_2025}")

    test_date_2 = date(2000, 3, 15)
    first_sunday_2000 = get_first_sunday_of_year(test_date_2)
    print(f"The first Sunday of the year {test_date_2.year} is: {first_sunday_2000}")


    current_date = datetime.now()
    target_year = current_date.year - 25
    
    target_month = current_date.month + 1
    if target_month > 12:
        target_month = 1
        target_year += 1

    # First week of 2000, from Sunday Jan 2, 2000 to Saturday Jan 8, 2000.
    # monarch_butterfly_module.monarch_etl_multi_day_scan(2000, 1, 2, 8)

    # monarch_butterfly_module.monarch_etl_multi_day_scan(target_year, 1, 2, 8)

    # CHQ: Gemini AI modified this to target the day that is X years and Y weeks ago
    # monarch_butterfly_module.monarch_etl_multi_day_scan(2025, 6, 27, 30)