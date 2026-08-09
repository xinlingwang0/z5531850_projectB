"""Reusable ETL and integrity checks for RiskBridge Funds Part B.

This module loads the provided project data through ``src.data_access`` and
creates clean, auditable price and headline panels for the Part B modelling
pipeline. Portfolio construction, FinVADER sentiment scoring, fusion, and app
logic live in their dedicated modules.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

import data_access


@dataclass
class IntegrityResult:
    """Record one data-quality check and how it was handled."""

    dataset: str
    check: str
    issue_count: int
    action: str
    notes: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "check": self.check,
            "issue_count": self.issue_count,
            "action": self.action,
            "notes": self.notes,
        }


def normalise_date(series: pd.Series) -> pd.Series:
    """Return timezone-naive, midnight-normalised dates.

    News dates are timezone-aware while price dates are usually timezone-naive.
    Normalising them before merges avoids dtype errors and silent mismatches.
    """
    dates = pd.to_datetime(series, errors="coerce")
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_convert(None)
    return dates.dt.normalize()


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all three provided datasets through the course helper."""
    equities = data_access.load_equity_prices()
    crypto = data_access.load_crypto_prices()
    headlines = data_access.load_news_headlines()
    return equities, crypto, headlines


def clean_price_panel(
    prices: pd.DataFrame,
    dataset_name: str,
    *,
    end_date: str | None = None,
    has_sector: bool = True,
) -> tuple[pd.DataFrame, list[IntegrityResult]]:
    """Clean an equity or crypto price panel and record checks.

    Price panels must be unique by ticker-date before returns are computed. Real
    extreme returns are documented later, not deleted here.
    """
    checks: list[IntegrityResult] = []
    df = prices.copy()

    required = {"ticker", "date", "adjClose"}
    missing_columns = required - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required columns: {sorted(missing_columns)}"
        )

    df["date"] = normalise_date(df["date"])
    invalid_dates = int(df["date"].isna().sum())
    if invalid_dates:
        df = df.loc[df["date"].notna()].copy()
    checks.append(
        IntegrityResult(
            dataset_name,
            "invalid or missing dates",
            invalid_dates,
            "dropped rows without a valid date" if invalid_dates else "no action needed",
            "A valid date is required for return construction and walk-forward timing.",
        )
    )

    if end_date is not None:
        before = len(df)
        df = df.loc[df["date"] <= pd.Timestamp(end_date)].copy()
        checks.append(
            IntegrityResult(
                dataset_name,
                "coverage cap",
                before - len(df),
                f"kept observations dated on or before {end_date}",
                "The crypto bundle includes stray 2024-01-01 observations; the project sample ends on 2023-12-31.",
            )
        )

    df["ticker"] = df["ticker"].fillna("").astype(str).str.strip()
    df["adjClose"] = pd.to_numeric(df["adjClose"], errors="coerce")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    duplicate_groups = int(df.duplicated(["ticker", "date"], keep=False).sum())
    before = len(df)
    if duplicate_groups:
        df = df.drop_duplicates(["ticker", "date"], keep="last").reset_index(drop=True)
    duplicate_rows_dropped = before - len(df)
    checks.append(
        IntegrityResult(
            dataset_name,
            "duplicate ticker-date rows",
            duplicate_rows_dropped,
            "kept last sorted row for duplicate ticker-date keys" if duplicate_rows_dropped else "no action needed",
            f"Price panels should be unique by ticker-date; {duplicate_groups} row(s) belonged to duplicate groups.",
        )
    )

    invalid_tickers = int(df["ticker"].eq("").sum())
    if invalid_tickers:
        df = df.loc[~df["ticker"].eq("")].copy()
    checks.append(
        IntegrityResult(
            dataset_name,
            "blank tickers",
            invalid_tickers,
            "dropped rows without a ticker" if invalid_tickers else "no action needed",
            "Ticker is required for panel returns and portfolio weights.",
        )
    )

    missing_adj = int(df["adjClose"].isna().sum())
    if missing_adj:
        df = df.loc[df["adjClose"].notna()].copy()
    checks.append(
        IntegrityResult(
            dataset_name,
            "missing adjusted-close values",
            missing_adj,
            "dropped rows without adjusted close" if missing_adj else "no action needed",
            "Adjusted close is required for return construction; prices are never forward-filled here.",
        )
    )

    nonpositive_adj = int((df["adjClose"] <= 0).sum())
    if nonpositive_adj:
        df = df.loc[df["adjClose"] > 0].copy()
    checks.append(
        IntegrityResult(
            dataset_name,
            "non-positive adjusted-close values",
            nonpositive_adj,
            "dropped non-positive adjusted-close rows" if nonpositive_adj else "no action needed",
            "Simple returns require strictly positive adjusted-close prices.",
        )
    )

    if has_sector:
        if "sector" not in df.columns:
            raise ValueError(f"{dataset_name} is missing the required sector column")
        df["sector"] = df["sector"].fillna("").astype(str).str.strip()
        missing_sector = int(df["sector"].eq("").sum())
        checks.append(
            IntegrityResult(
                dataset_name,
                "missing sector labels",
                missing_sector,
                "kept rows; sector labels checked for reporting coverage" if missing_sector else "no action needed",
                "Sector labels are needed for sector-level headline and reporting summaries.",
            )
        )

    return df.reset_index(drop=True), checks


def clean_headlines(headlines: pd.DataFrame) -> tuple[pd.DataFrame, list[IntegrityResult]]:
    """Clean headline data and remove exact duplicate titles.

    Multiple rows per ticker-date are valid news flow, so duplicates are defined
    on ticker-date-title rather than ticker-date alone.
    """
    checks: list[IntegrityResult] = []
    df = headlines.copy()

    required = {"date", "ticker", "sector", "title"}
    missing_columns = required - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"news_headlines is missing required columns: {sorted(missing_columns)}"
        )

    df["date"] = normalise_date(df["date"])
    for col in ["ticker", "sector", "title", "publisher", "url"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).str.strip()

    duplicate_mask = df.duplicated(["ticker", "date", "title"], keep="first")
    duplicate_rows_dropped = int(duplicate_mask.sum())
    df = df.loc[~duplicate_mask].copy().reset_index(drop=True)
    checks.append(
        IntegrityResult(
            "news_headlines",
            "exact duplicate headlines",
            duplicate_rows_dropped,
            "dropped exact duplicates on ticker-date-title" if duplicate_rows_dropped else "no action needed",
            "Multiple headlines for the same ticker-date can be valid; exact repeated titles are not counted twice.",
        )
    )

    invalid_keys = df["date"].isna() | df["ticker"].eq("") | df["sector"].eq("")
    invalid_key_rows = int(invalid_keys.sum())
    if invalid_key_rows:
        df = df.loc[~invalid_keys].copy()
    checks.append(
        IntegrityResult(
            "news_headlines",
            "invalid date, ticker, or sector keys",
            invalid_key_rows,
            "dropped rows with invalid alignment keys" if invalid_key_rows else "no action needed",
            "Ticker, sector, and date are required for trading-day and sector aggregation.",
        )
    )

    blank_titles = int(df["title"].eq("").sum())
    if blank_titles:
        df = df.loc[~df["title"].eq("")].copy()
    checks.append(
        IntegrityResult(
            "news_headlines",
            "blank titles",
            blank_titles,
            "dropped blank titles before sentiment scoring" if blank_titles else "no action needed",
            "FinVADER requires non-empty headline text; excluded rows remain quantified in this audit.",
        )
    )
    df["text_raw"] = df["title"]
    return df.reset_index(drop=True), checks


def price_calendar_audit(
    prices: pd.DataFrame, dataset_name: str, *, calendar: str
) -> IntegrityResult:
    """Audit ticker-date coverage against the relevant calendar."""
    if prices.empty:
        return IntegrityResult(
            dataset_name, "calendar coverage", 0, "no data", "No rows loaded."
        )

    tickers = prices["ticker"].nunique()
    start, end = prices["date"].min(), prices["date"].max()
    if calendar == "equity":
        expected_dates = pd.Index(sorted(prices["date"].dropna().unique()))
        notes = (
            "Equity coverage is checked against observed equity trading dates; "
            "market holidays are not treated as missing."
        )
    elif calendar == "crypto":
        expected_dates = pd.date_range(start, end, freq="D")
        notes = (
            "Crypto is checked against a daily seven-day calendar before later "
            "alignment to equity dates."
        )
    else:
        raise ValueError("calendar must be 'equity' or 'crypto'")

    expected = len(expected_dates) * tickers
    observed = prices.drop_duplicates(["ticker", "date"]).shape[0]
    missing = max(int(expected - observed), 0)
    return IntegrityResult(
        dataset_name,
        "missing ticker-date observations",
        missing,
        "documented and retained clean panel",
        notes,
    )


def load_clean_equities() -> pd.DataFrame:
    """Load and clean equity prices for modelling."""
    df, _ = clean_price_panel(
        data_access.load_equity_prices(), "equity_prices", has_sector=True
    )
    return df


def load_clean_crypto() -> pd.DataFrame:
    """Load and clean crypto prices, capped to the project sample."""
    df, _ = clean_price_panel(
        data_access.load_crypto_prices(),
        "crypto_prices",
        end_date="2023-12-31",
        has_sector=False,
    )
    return df


def make_dataset_inventory(
    equities_raw: pd.DataFrame,
    equities: pd.DataFrame,
    crypto_raw: pd.DataFrame,
    crypto: pd.DataFrame,
    news_raw: pd.DataFrame,
    news: pd.DataFrame,
) -> pd.DataFrame:
    """Create a dataset-inventory table for the reusable data foundation."""

    def row(
        name: str,
        raw: pd.DataFrame,
        clean: pd.DataFrame,
        frequency: str,
        coverage: str,
        source: str,
    ) -> dict[str, Any]:
        tickers = clean["ticker"].nunique() if "ticker" in clean else ""
        sectors = clean["sector"].nunique() if "sector" in clean else ""
        return {
            "dataset": name,
            "rows_raw": len(raw),
            "rows_clean": len(clean),
            "tickers_or_assets": tickers,
            "sectors": sectors,
            "start_date": clean["date"].min().date().isoformat() if len(clean) else "",
            "end_date": clean["date"].max().date().isoformat() if len(clean) else "",
            "frequency": frequency,
            "coverage": coverage,
            "source_or_provenance": source,
        }

    return pd.DataFrame(
        [
            row(
                "equity_prices",
                equities_raw,
                equities,
                "daily equity trading days",
                "50 US equities across 10 sectors; OHLCV plus adjusted close",
                "Loaded through src.data_access.load_equity_prices(); cleaned by ticker-date key.",
            ),
            row(
                "crypto_prices",
                crypto_raw,
                crypto,
                "daily seven-day calendar",
                "10 cryptocurrencies; price-only diversifiers; returns computed before equity-calendar alignment",
                "Loaded through src.data_access.load_crypto_prices(); capped at 2023-12-31 before return construction.",
            ),
            row(
                "news_headlines",
                news_raw,
                news,
                "daily headline records aligned to equity dates later",
                "Headline-only news for the 50 equities; ticker, sector, title, URL and publisher fields",
                "Loaded through src.data_access.load_news_headlines(); exact duplicates removed on ticker-date-title.",
            ),
        ]
    )


def integrity_table(results: list[IntegrityResult]) -> pd.DataFrame:
    """Convert integrity records to a DataFrame."""
    return pd.DataFrame([r.as_dict() for r in results])
