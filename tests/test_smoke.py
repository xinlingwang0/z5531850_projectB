"""Smoke test: imports resolve and the data loads.

    python tests/test_smoke.py
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src import data_access  # noqa: E402


def test_imports():
    assert hasattr(data_access, "load_equity_prices")


def test_data_loads():
    eq = data_access.load_equity_prices()
    assert eq.shape[0] > 0
    assert {"ticker", "date", "adjClose", "sector"}.issubset(eq.columns)


if __name__ == "__main__":
    test_imports()
    print("imports OK")
    try:
        test_data_loads()
        print("data load OK")
    except Exception as e:
        print("data load skipped/failed (need network or FINS_DATA_ZIP):", e)
