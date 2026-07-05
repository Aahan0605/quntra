#!/usr/bin/env python3
"""Verify all QuNtra runtime dependencies import cleanly (Task 1-1)."""
import importlib
import sys

# (import name, pip name, required)
PACKAGES = [
    ("numpy", "numpy", True),
    ("pandas", "pandas", True),
    ("sklearn", "scikit-learn", True),
    ("xgboost", "xgboost", True),
    ("sqlalchemy", "sqlalchemy", True),
    ("alembic", "alembic", True),
    ("apscheduler", "apscheduler", True),
    ("pytz", "pytz", True),
    ("dotenv", "python-dotenv", True),
    ("pandas_market_calendars", "pandas-market-calendars", True),
    ("yfinance", "yfinance", True),
    ("jugaad_data", "jugaad-data", True),
    ("telegram", "python-telegram-bot", True),
    ("kiteconnect", "kiteconnect", True),
    ("ta", "ta", True),  # pandas-ta is unavailable (pulled from PyPI+GitHub); MIT 'ta' replaces it
    ("pyfolio", "pyfolio-reloaded", True),
    ("mlflow", "mlflow-skinny", True),
    ("Fundamentals", "Bharat-SM-Data", True),   # exposes Base/Derivatives/Fundamentals/Technical
    ("Derivatives", "Bharat-SM-Data", True),
    ("asyncpg", "asyncpg", True),
    ("psycopg2", "psycopg2-binary", True),
    ("cvxpy", "cvxpy", True),
    ("qiskit", "qiskit", True),
    ("vectorbt", "vectorbt", False),
]


def main() -> int:
    hard_fail = False
    for mod, pip_name, required in PACKAGES:
        try:
            importlib.import_module(mod)
            print(f"OK       {pip_name}")
        except Exception as e:  # noqa: BLE001
            tag = "FAIL    " if required else "OPTIONAL"
            print(f"{tag} {pip_name} — {type(e).__name__}: {e}")
            if required:
                hard_fail = True
    print("\nAll required imports OK" if not hard_fail
          else "\nRequired imports FAILED")
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
