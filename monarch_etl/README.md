# monarch_etl – Module Map

## File structure

```
monarch_etl/
├── __init__.py          # re-exports the three public ETL entry points
├── config.py            # env vars, constants, dtype maps, column lists
├── logger.py            # shared logger instance
├── retry_config.py      # tenacity retry decorator (shared by all HTTP clients)
│
├── gbif_client.py       # GBIF API: single-page fetch + multi-page extraction
├── geocode_client.py    # Reverse-geocoding API: lat/lon → county + city
│
├── cleaning.py          # DataFrame cleaning (dates, coordinates, counts)
├── enrichment.py        # Attaches geocoding + time_only columns
├── transform.py         # Orchestrates clean → enrich → schema
│
├── schema.py            # Canonical column list + SQLAlchemy dtype map
├── table_naming.py      # Derives DB table names from dates
│
├── db_loader.py         # Writes a DataFrame to PostgreSQL
├── inventory.py         # Manages the data_inventory tracking table
│
└── etl.py               # High-level orchestration (the public API)
```

## Dependency graph

```
etl.py
  ├── gbif_client.py   →  retry_config, config, logger
  ├── transform.py
  │     ├── cleaning.py     →  logger
  │     ├── enrichment.py   →  geocode_client → retry_config, config, logger
  │     └── schema.py       →  config
  ├── db_loader.py     →  schema, logger
  ├── inventory.py     →  logger
  └── table_naming.py  →  config
```

## Usage

```python
from etl import monarch_etl, monarch_etl_day_scan, monarch_etl_multi_day_scan

CONN = "postgresql+psycopg2://user:pw@host:5432/dbname"

# Whole month
monarch_etl(2025, 6, CONN)

# Single day
monarch_etl_day_scan(2025, 6, 15, CONN)

# Range of days
monarch_etl_multi_day_scan(2025, 6, 1, 30, CONN)
```

## Environment variables required

| Variable                    | Description                       |
| --------------------------- | --------------------------------- |
| `NEON_DB_HOST`              | PostgreSQL host                   |
| `NEON_DB_NAME`              | Database name                     |
| `NEON_DB_USER`              | Database user                     |
| `NEON_DB_PASSWORD`          | Database password                 |
| `NEON_DB_PORT`              | Port (default `5432`)             |
| `REVERSE_GEOCACHE_API_BASE` | Base URL of the geocoding API     |
| `REVERSE_GEOCACHE_API_KEY`  | API key for the geocoding service |
