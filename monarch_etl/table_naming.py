# CHQ: Claude AI generated file

"""
table_naming.py
---------------
Pure utility: derives the PostgreSQL table name for a given date.

Keeping this logic isolated means any future naming-convention change
is a one-line edit in one place.

Examples
--------
>>> table_name_for_day(2025, 6, 1)
'june012025'
>>> table_name_for_day(2025, 6, 10)
'june102025'
>>> table_name_for_month(2025, 6)
'june 2025'
"""

from config import MONTH_NAMES


def table_name_for_day(year: int, month: int, day: int) -> str:
    """
    Return the table name for a single-day load.

    Format: ``<monthname><zero-padded-day><year>``
    e.g. ``june012025``, ``december312021``
    """
    month_str = MONTH_NAMES[month]
    day_str = f"{day:02d}"  # zero-pad to 2 digits
    return f"{month_str}{day_str}{year}"


def table_name_for_month(year: int, month: int) -> str:
    """
    Return the table name for a whole-month load.

    Format: ``<monthname> <year>``
    e.g. ``june 2025``
    """
    return f"{MONTH_NAMES[month]} {year}"
    