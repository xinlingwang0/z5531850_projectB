# PROVIDED - do not edit. Frozen copy of the course data_access helper.
"""Single entry point for loading the hosted project datasets.

The data is published as ONE public ZIP (WordPress blocks raw .parquet), holding
the parquet files. We download + unzip ONCE (cached) and read each file from
memory. Students and the Streamlit app call these functions; nothing else touches
the network. Switch host by changing one setting:

    set FINS_DATA_ZIP=https://your-site/path/project_data.zip   (env var)
    # or point it at a local .zip for offline testing

Caching uses Streamlit's cache inside Streamlit and an in-process LRU otherwise,
so the same code is fast on a student laptop and on Streamlit Community Cloud.
"""
from __future__ import annotations

import os
import io
import zipfile
import functools

import pandas as pd
import requests

DATA_ZIP_URL = os.environ.get(
    "FINS_DATA_ZIP",
    "https://drive.google.com/uc?export=download&id=1h0Wy12_qgR_NZJqtSxI9LwPEVKgp5DzH",
)
# Backup host, used automatically if the primary fails or is not a valid ZIP.
DATA_ZIP_BACKUP_URL = "https://openbondassetpricing.com/wp-content/uploads/2026/06/project_data.zip"


def _cache(func):
    """Use st.cache_data inside Streamlit, else a plain LRU cache."""
    try:
        import streamlit as st
        return st.cache_data(ttl=86_400, show_spinner=False)(func)
    except Exception:
        return functools.lru_cache(maxsize=8)(func)


def _fetch(url: str) -> bytes:
    if os.path.exists(url):
        return open(url, "rb").read()
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return resp.content


@_cache
def _bundle() -> dict:
    """Download + unzip the data bundle once; try the primary URL, then the backup."""
    urls = [DATA_ZIP_URL]
    if DATA_ZIP_BACKUP_URL and DATA_ZIP_BACKUP_URL != DATA_ZIP_URL:
        urls.append(DATA_ZIP_BACKUP_URL)
    last_err = None
    for url in urls:
        try:
            z = zipfile.ZipFile(io.BytesIO(_fetch(url)))
            return {os.path.basename(n): z.read(n) for n in z.namelist() if not n.endswith("/")}
        except Exception as exc:  # network error, or not a valid ZIP (e.g. an interstitial page)
            last_err = exc
    raise RuntimeError(f"could not load data bundle from {urls}: {last_err}")


def _read_parquet(name: str) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(_bundle()[name]))


@_cache
def load_equity_prices() -> pd.DataFrame:
    """50 US equities, daily OHLCV + adjClose + sector (2020-2023)."""
    df = _read_parquet("equity_prices.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df


@_cache
def load_crypto_prices() -> pd.DataFrame:
    """10 cryptocurrencies, daily OHLCV + adjClose (2020-2023, 365-day calendar)."""
    df = _read_parquet("crypto_prices.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df


@_cache
def load_news_headlines() -> pd.DataFrame:
    """Headlines only: date, ticker, sector, title, url, publisher (2020-2023)."""
    df = _read_parquet("news_headlines.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df


@_cache
def load_sector_universe() -> pd.DataFrame:
    """Ticker -> sector map. Derived from the equity prices (no separate file needed)."""
    eq = load_equity_prices()
    return (eq[["ticker", "sector"]].drop_duplicates()
              .sort_values(["sector", "ticker"]).reset_index(drop=True))


__all__ = [
    "DATA_ZIP_URL", "load_equity_prices", "load_crypto_prices",
    "load_news_headlines", "load_sector_universe",
]
