"""Return features and headline assembly for RiskBridge Funds Part B.

These functions prepare the price and text inputs consumed by the Part B
portfolio, FinVADER sentiment, and fusion modules.
"""
from __future__ import annotations

import re
import warnings
from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd


STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "into", "over", "under",
    "after", "before", "amid", "will", "says", "say", "said", "new", "its", "are",
    "has", "have", "had", "was", "were", "been", "more", "less", "than", "about",
    "how", "why", "what", "when", "where", "your", "you", "our", "their", "his",
    "her", "not", "but", "out", "off", "all", "can", "may", "could", "would",
    "should", "stock", "stocks", "market", "markets", "company", "companies",
    "shares", "share", "inc", "corp", "ltd", "plc", "nasdaq", "nyse", "s", "u",
    "us",
}

RISK_TERMS = {
    "risk", "risks", "fall", "falls", "fell", "drop", "drops", "dropped", "down",
    "loss", "losses", "warning", "warns", "cut", "cuts", "miss", "misses",
    "slump", "plunge", "rally", "gain", "gains", "rise", "rises", "up", "surge",
    "beats", "beat", "growth", "earnings", "inflation", "rates", "rate", "oil",
    "energy", "bitcoin", "crypto",
}


def _normalise_date(series: pd.Series) -> pd.Series:
    dates = pd.to_datetime(series, errors="coerce")
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_convert(None)
    return dates.dt.normalize()


def daily_returns(
    prices: pd.DataFrame,
    price_col: str = "adjClose",
    asset_class: str | None = None,
) -> pd.DataFrame:
    """Compute simple daily returns by ticker from adjusted-close prices.

    The function computes returns within each ticker on its native calendar. For
    crypto this must happen before any alignment to the equity trading calendar.
    """
    required = {"ticker", "date", price_col}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"price panel is missing required columns: {sorted(missing)}")

    cols = ["ticker", "date", price_col]
    if "sector" in prices.columns:
        cols.append("sector")
    df = prices[cols].copy()
    df["ticker"] = df["ticker"].fillna("").astype(str).str.strip()
    df["date"] = _normalise_date(df["date"])
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=["ticker", "date", price_col])
    df = df.loc[~df["ticker"].eq("")].copy()
    duplicate_keys = df.duplicated(["ticker", "date"], keep=False)
    if duplicate_keys.any():
        examples = (
            df.loc[duplicate_keys, ["ticker", "date"]]
            .drop_duplicates()
            .head(5)
            .to_dict("records")
        )
        raise ValueError(
            "price panel has duplicate ticker-date rows before return "
            f"construction; clean it with src.etl.clean_price_panel first. "
            f"Examples: {examples}"
        )
    df = df.sort_values(["ticker", "date"])
    df["return"] = df.groupby("ticker", sort=False)[price_col].pct_change(
        fill_method=None
    )
    df = df.dropna(subset=["return"]).reset_index(drop=True)
    if asset_class is not None:
        df.insert(0, "asset_class", asset_class)
    return df


def wide_returns(returns: pd.DataFrame, value_col: str = "return") -> pd.DataFrame:
    """Convert a long return panel into a date-by-ticker wide panel."""
    required = {"ticker", "date", value_col}
    missing = required - set(returns.columns)
    if missing:
        raise ValueError(f"return panel is missing required columns: {sorted(missing)}")
    df = returns.copy()
    df["ticker"] = df["ticker"].fillna("").astype(str).str.strip()
    df["date"] = _normalise_date(df["date"])
    df = df.dropna(subset=["ticker", "date", value_col])
    df = df.loc[~df["ticker"].eq("")].copy()
    duplicate_keys = df.duplicated(["date", "ticker"], keep=False)
    if duplicate_keys.any():
        examples = (
            df.loc[duplicate_keys, ["date", "ticker"]]
            .drop_duplicates()
            .head(5)
            .to_dict("records")
        )
        raise ValueError(
            "return panel has duplicate date-ticker rows and cannot be safely "
            f"pivoted to wide form. Examples: {examples}"
        )
    wide = df.pivot(index="date", columns="ticker", values=value_col).sort_index()
    wide.index.name = "date"
    return wide


def align_crypto_to_equity_calendar(
    crypto_returns: pd.DataFrame,
    equity_dates: Iterable[pd.Timestamp],
    value_col: str = "return",
) -> pd.DataFrame:
    """Left-align native daily crypto returns to equity trading dates.

    Crypto returns must already have been computed on the native seven-day
    calendar. Reindexing those returns to the equity calendar intentionally
    excludes weekend-only rows, as required by the project brief. It does not
    recompute Friday-to-Monday returns from price levels.
    """
    required = {"ticker", "date", value_col}
    missing = required - set(crypto_returns.columns)
    if missing:
        raise ValueError(
            f"crypto return panel is missing required columns: {sorted(missing)}"
        )

    equity_index = pd.DatetimeIndex(_normalise_date(pd.Series(list(equity_dates))))
    equity_index = equity_index.dropna().sort_values().unique()
    native_wide = wide_returns(crypto_returns, value_col=value_col)
    aligned = native_wide.reindex(equity_index)
    aligned.index.name = "date"
    return aligned


def combined_returns_panel(
    equity_returns: pd.DataFrame,
    crypto_equity_calendar_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Create a combined equity-plus-crypto return panel on equity dates."""
    eq_wide = wide_returns(equity_returns)
    cr_aligned = crypto_equity_calendar_returns.reindex(eq_wide.index)
    combined = pd.concat([eq_wide, cr_aligned], axis=1).sort_index()
    combined.index.name = "date"
    return combined


def return_descriptive_stats(
    equity_returns: pd.DataFrame, crypto_returns: pd.DataFrame
) -> pd.DataFrame:
    """Summarise daily returns by asset class."""
    rows = []
    for label, df, annual_factor in [
        ("Equities", equity_returns, 252),
        ("Crypto", crypto_returns, 365),
    ]:
        r = df["return"].dropna()
        rows.append(
            {
                "asset_class": label,
                "n_observations": int(r.shape[0]),
                "n_assets": int(df["ticker"].nunique()),
                "mean_daily_return": r.mean(),
                "annualised_return_arithmetic": r.mean() * annual_factor,
                "daily_volatility": r.std(),
                "annualised_volatility": r.std() * np.sqrt(annual_factor),
                "min_daily_return": r.min(),
                "max_daily_return": r.max(),
                "skewness": r.skew(),
                "kurtosis": r.kurtosis(),
                "annualisation_factor": annual_factor,
            }
        )
    return pd.DataFrame(rows)


def top_return_outliers(
    equity_returns: pd.DataFrame, crypto_returns: pd.DataFrame, n: int = 25
) -> pd.DataFrame:
    """Return the largest absolute daily returns across both asset classes."""
    cols = ["asset_class", "ticker", "date", "return"]
    eq = equity_returns.copy()
    cr = crypto_returns.copy()
    if "asset_class" not in eq:
        eq.insert(0, "asset_class", "Equity")
    if "asset_class" not in cr:
        cr.insert(0, "asset_class", "Crypto")
    combined = pd.concat([eq[cols], cr[cols]], ignore_index=True)
    combined["absolute_return"] = combined["return"].abs()
    out = combined.sort_values("absolute_return", ascending=False).head(n).copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date.astype(str)
    return out.reset_index(drop=True)


def tail_risk_summary(
    equity_returns: pd.DataFrame, crypto_returns: pd.DataFrame
) -> pd.DataFrame:
    """Compare return-tail behaviour by asset class."""
    rows = []
    for label, df in [("Equities", equity_returns), ("Crypto", crypto_returns)]:
        r = df["return"].dropna()
        rows.append(
            {
                "asset_class": label,
                "n_returns": int(r.shape[0]),
                "count_abs_return_gt_5pct": int((r.abs() > 0.05).sum()),
                "count_abs_return_gt_10pct": int((r.abs() > 0.10).sum()),
                "count_abs_return_gt_20pct": int((r.abs() > 0.20).sum()),
                "share_abs_return_gt_5pct": (r.abs() > 0.05).mean(),
                "share_abs_return_gt_10pct": (r.abs() > 0.10).mean(),
                "share_abs_return_gt_20pct": (r.abs() > 0.20).mean(),
                "p01_daily_return": r.quantile(0.01),
                "p05_daily_return": r.quantile(0.05),
                "p95_daily_return": r.quantile(0.95),
                "p99_daily_return": r.quantile(0.99),
                "worst_daily_return": r.min(),
                "best_daily_return": r.max(),
            }
        )
    return pd.DataFrame(rows)


def calendar_readiness_summary(
    equities: pd.DataFrame,
    crypto: pd.DataFrame,
    eq_returns: pd.DataFrame,
    cr_returns: pd.DataFrame,
    cr_aligned: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise calendar differences relevant to Part B model inputs."""
    equity_dates = pd.Index(_normalise_date(eq_returns["date"]).drop_duplicates()).sort_values()
    crypto_dates = pd.Index(_normalise_date(cr_returns["date"]).drop_duplicates()).sort_values()
    return pd.DataFrame(
        [
            {
                "panel": "equity_prices",
                "native_calendar": "equity trading days",
                "native_dates": int(equity_dates.nunique()),
                "tickers_or_assets": int(equities["ticker"].nunique()),
                "return_observations_native": int(len(eq_returns)),
                "dates_after_equity_alignment": int(equity_dates.nunique()),
                "notes": "This is the anchor calendar for combined Part B modelling.",
            },
            {
                "panel": "crypto_prices",
                "native_calendar": "seven-day calendar",
                "native_dates": int(crypto_dates.nunique()),
                "tickers_or_assets": int(crypto["ticker"].nunique()),
                "return_observations_native": int(len(cr_returns)),
                "dates_after_equity_alignment": int(cr_aligned.shape[0]),
                "notes": "Native crypto daily returns are used for the crypto fund; the combined panel left-aligns those returns to equity dates, so weekend-only rows are excluded.",
            },
        ]
    )


def _map_to_next_trading_day(dates: pd.Series, trading_dates: pd.Index) -> pd.Series:
    """Map each date to the same or next available equity trading date."""
    trading = pd.DatetimeIndex(trading_dates).sort_values().unique()
    raw = _normalise_date(dates).to_numpy(dtype="datetime64[ns]")
    trading_np = trading.to_numpy(dtype="datetime64[ns]")
    positions = np.searchsorted(trading_np, raw, side="left")
    mapped = np.full(len(raw), np.datetime64("NaT"), dtype="datetime64[ns]")
    valid = positions < len(trading_np)
    mapped[valid] = trading_np[positions[valid]]
    return pd.Series(pd.to_datetime(mapped), index=dates.index, name="trading_date")


def assemble_headline_panel(
    headlines: pd.DataFrame, equity_trading_dates: Iterable[pd.Timestamp]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align individual headlines and assemble a daily ticker-sector panel.

    FinVADER should score the individual rows in the first returned DataFrame.
    The second DataFrame is an aggregation for coverage and display, not the
    sentiment-model input.
    """
    required = {"date", "ticker", "sector", "title"}
    missing = required - set(headlines.columns)
    if missing:
        raise ValueError(f"headline panel is missing required columns: {sorted(missing)}")

    df = headlines.copy()
    trading_dates = pd.Index(_normalise_date(pd.Series(list(equity_trading_dates))))
    trading_dates = trading_dates.dropna().sort_values().unique()
    if len(trading_dates) == 0:
        raise ValueError("equity_trading_dates must contain at least one date")

    for col in ["ticker", "sector", "title", "text_raw", "publisher"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).str.strip()

    df["date"] = _normalise_date(df["date"])
    df["trading_date"] = _map_to_next_trading_day(df["date"], trading_dates)
    unmapped = int(df["trading_date"].isna().sum())
    if unmapped:
        warnings.warn(
            f"{unmapped} headline(s) fall after the final available equity "
            "trading date and were excluded from the aligned panel.",
            RuntimeWarning,
            stacklevel=2,
        )
    df = df.dropna(subset=["trading_date"]).copy()
    df["trading_date"] = pd.to_datetime(df["trading_date"]).dt.normalize()
    text_col = "text_raw" if df["text_raw"].ne("").any() else "title"
    grouped = (
        df.groupby(["trading_date", "ticker", "sector"], as_index=False)
        .agg(
            headline_count=(text_col, "size"),
            titles=(text_col, lambda x: " || ".join(x.astype(str))),
            unique_publishers=("publisher", lambda x: x.replace("", np.nan).nunique()),
        )
        .sort_values(["trading_date", "sector", "ticker"])
    )
    return df.reset_index(drop=True), grouped.reset_index(drop=True)


def news_coverage_by_sector(aligned_headlines: pd.DataFrame) -> pd.DataFrame:
    """Summarise headline coverage by sector."""
    required = {"sector", "ticker", "trading_date", "title"}
    missing = required - set(aligned_headlines.columns)
    if missing:
        raise ValueError(
            f"aligned headline panel is missing required columns: {sorted(missing)}"
        )
    df = aligned_headlines.copy()
    if "publisher" not in df.columns:
        df["publisher"] = ""
    total = max(len(df), 1)
    trading_days = max(df["trading_date"].nunique(), 1)
    out = (
        df.groupby("sector")
        .agg(
            headline_count=("title", "size"),
            ticker_count=("ticker", "nunique"),
            trading_days_with_news=("trading_date", "nunique"),
            publisher_count=("publisher", lambda x: x.replace("", np.nan).nunique()),
        )
        .reset_index()
    )
    out["share_of_all_headlines"] = out["headline_count"] / total
    out["avg_headlines_per_trading_day"] = out["headline_count"] / trading_days
    return out.sort_values("headline_count", ascending=False).reset_index(drop=True)


def top_terms(
    headlines: pd.DataFrame, top_n: int = 30, *, risk_only: bool = False
) -> pd.DataFrame:
    """Count common terms in headline titles without assigning sentiment scores."""
    if "title" not in headlines.columns:
        raise ValueError("headline panel is missing required column: 'title'")
    counter: Counter[str] = Counter()
    for title in headlines["title"].dropna().astype(str):
        tokens = re.findall(r"[A-Za-z][A-Za-z\-']{2,}", title.lower())
        for token in tokens:
            if token in STOPWORDS or len(token) <= 2:
                continue
            if risk_only and token not in RISK_TERMS:
                continue
            counter[token] += 1
    rows = [{"term": term, "count": count} for term, count in counter.most_common(top_n)]
    return pd.DataFrame(rows)
