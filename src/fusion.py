"""Station 3 (extension) - fuse extended-lexicon sentiment into a fund.

Applies a bounded, transparent sentiment tilt to the target weights of the
existing ``Equity-Only Minimum Variance`` fund and prices the tilted schedule
with the same shared executor used by every other fund in the project
(``portfolios.run_backtest_from_target_schedule``), so the tilted and untilted
books stay directly comparable on rebalance timing, weight drift, turnover and
transaction costs.

Scope and design choices worth stating explicitly:

* Only the frozen extended lexicon is used here. Whether the extended lexicon
  differs from the baseline lexicon is a question the sentiment module already
  answers; repeating that comparison at the portfolio level would add a second
  set of results without adding evidence.
* Tilt strength is the fixed, symmetric pair ``lambda = +1`` (momentum) and
  ``lambda = -1`` (contrarian), plus the untilted base at ``lambda = 0``. Two
  equal-magnitude directions answer "does this signal help, hurt, or do
  nothing, and which way round" without the overfitting risk of searching a
  wider grid on the same out-of-sample window. Neither direction is selected
  after seeing the results, and neither is dropped for underperforming.
* A missing sentiment signal produces a tilt multiplier of exactly 1, leaving
  the weight untouched. That encodes "no information is available for this
  stock today", which is a different statement from "this stock's sentiment
  was measured and found to be neutral".
* Every variant is evaluated over the full 2021-2023 out-of-sample window with
  the project's single transaction-cost assumption. There is no in-sample
  tuning step and no tuning/holdout split.
"""
from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

# Reused, not reimplemented: one prior-history standardisation definition and
# one application-period guard for the whole project.
from .sentiment import assert_application_period_only, expanding_standardize

TICKER_Z_COLUMNS = ["date", "ticker", "extended_z_expanding"]

TILT_WEIGHT_COLUMNS = [
    "base_weight",
    "tilt_multiplier",
    "pre_normalisation_weight",
    "target_weight",
    "clipped_to_zero",
    "signal_available",
    "fallback_used",
]

WEIGHT_TOLERANCE = 1e-6


# ============================================================================
# Ticker-level extended sentiment signal
# ============================================================================


def build_ticker_extended_z(
    ticker_day: pd.DataFrame,
    trading_dates: Iterable[pd.Timestamp],
    tickers: Iterable[str],
    min_periods: int = 60,
) -> pd.DataFrame:
    """Ticker-level prior-history expanding z-scores for the extended lexicon.

    Standardises each ticker's extended sentiment against its own history with
    the same ``expanding_standardize`` definition the sector index already
    uses, then reindexes onto the complete equity trading calendar x the
    project's formal 50-stock universe. A ticker-day with no news is present as
    an explicit NaN row rather than silently absent, and nothing is forward
    filled, back filled, or set to zero: a missing z-score means the signal is
    unavailable, not that sentiment was neutral.
    """
    required = {"date", "ticker", "extended_compound"}
    missing = required - set(ticker_day.columns)
    if missing:
        raise ValueError(f"ticker_day is missing required columns: {sorted(missing)}")

    work = ticker_day[["date", "ticker", "extended_compound"]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    work["ticker"] = work["ticker"].astype(str)
    if work.duplicated(["date", "ticker"]).any():
        raise ValueError("ticker_day must be unique by (date, ticker)")

    work["extended_z_expanding"] = expanding_standardize(
        work, "extended_compound", group_col="ticker", min_periods=min_periods
    )

    grid_dates = pd.DatetimeIndex(
        sorted(pd.to_datetime(pd.Series(list(trading_dates))).dt.normalize().unique())
    )
    grid_tickers = sorted({str(ticker) for ticker in tickers})
    full_index = pd.MultiIndex.from_product(
        [grid_dates, grid_tickers], names=["date", "ticker"]
    )

    indexed = work.set_index(["date", "ticker"])[["extended_z_expanding"]]
    # Reindexing silently drops anything outside the grid, which would be real
    # data loss rather than a formatting detail - fail loudly instead.
    outside = indexed.index.difference(full_index)
    if len(outside) > 0:
        raise ValueError(
            f"{len(outside)} ticker-day observation(s) fall outside the "
            f"trading-date x ticker grid, e.g. {list(outside[:5])}"
        )

    result = indexed.reindex(full_index).reset_index()
    assert_application_period_only(result["date"], context="ticker_sentiment_z")
    return result[TICKER_Z_COLUMNS]


# ============================================================================
# Signal timing: which trading day's sentiment a target weight may use
# ============================================================================


def previous_trading_day(
    effective_date: pd.Timestamp, trading_dates: Iterable[pd.Timestamp]
) -> pd.Timestamp:
    """Return the latest trading day strictly before ``effective_date``.

    This is the fund's ``signal_date``: the most recent day whose sentiment was
    fully observable before the new target weights start being held. It is
    derived from the trading calendar rather than assumed equal to the
    portfolio's formation date, so the timing stays correct even if the two
    ever diverge.
    """
    calendar = pd.DatetimeIndex(
        sorted(pd.to_datetime(pd.Series(list(trading_dates))).dt.normalize().unique())
    )
    effective_date = pd.Timestamp(effective_date).normalize()
    earlier = calendar[calendar < effective_date]
    if len(earlier) == 0:
        raise ValueError(
            f"no trading day strictly before {effective_date.date()} is available "
            "in the supplied calendar"
        )
    return pd.Timestamp(earlier[-1])


def lookup_ticker_signal(
    ticker_z: pd.DataFrame, signal_date: pd.Timestamp, tickers: Iterable[str]
) -> pd.Series:
    """Exact ``(signal_date, ticker)`` extended z-score lookup - nothing else.

    Returns the z-score recorded for that exact ticker on that exact trading
    day, or NaN when the row is absent or its z-score is missing. There is
    deliberately no as-of match, no search backwards for the most recent
    non-missing value, no forward or backward fill and no rolling average: each
    of those would either build a target from something other than the latest
    fully-observed signal, or resurrect a stale reading and present it as
    current information.
    """
    signal_date = pd.Timestamp(signal_date).normalize()
    ticker_index = pd.Index([str(ticker) for ticker in tickers], name="ticker")
    if ticker_index.has_duplicates:
        raise ValueError("tickers must be unique")

    same_day = ticker_z.loc[
        pd.to_datetime(ticker_z["date"]).dt.normalize() == signal_date
    ]
    if same_day.duplicated("ticker").any():
        raise ValueError(f"ticker_z has duplicate tickers on {signal_date.date()}")

    values = same_day.set_index("ticker")["extended_z_expanding"]
    return values.reindex(ticker_index).astype(float)


# ============================================================================
# The tilt itself: multiplier, then weights
# ============================================================================


def sentiment_tilt_multiplier(z: pd.Series, lam: float) -> pd.Series:
    """Return ``1 + lam * z``, with a multiplier of exactly 1 where z is missing.

    A multiplier of 1 leaves the base weight untouched. It is used for a
    missing z-score because no sentiment information is available for that
    stock on that signal date - it is NOT a claim that the stock's sentiment
    was measured and found to be neutral. Treating "unavailable" as "neutral"
    would let absent data quietly influence the portfolio.
    """
    if not isinstance(z, pd.Series):
        raise TypeError("z must be a pandas Series indexed by ticker")
    if not np.isfinite(float(lam)):
        raise ValueError("lam must be finite")

    multiplier = 1.0 + float(lam) * z.astype(float)
    return multiplier.where(z.notna(), 1.0)


def apply_sentiment_tilt(
    base_weights: pd.Series,
    multiplier: pd.Series,
    *,
    signal_available: pd.Series | None = None,
) -> pd.DataFrame:
    """Apply a tilt multiplier to base weights, then clip and renormalise.

    Steps, in order:

    1. ``pre_normalisation_weight = base_weight * tilt_multiplier``.
    2. Clip negative pre-normalisation weights to zero (recorded per ticker),
       keeping the fund long-only like every other fund in the project.
    3. Renormalise the clipped weights so the target weights sum to one.
    4. If clipping removed the entire book, fall back to the untilted base
       weights, warn, and flag ``fallback_used`` - the alternative would be a
       division by zero or an all-NaN target.

    A ticker with no signal keeps ``pre_normalisation_weight == base_weight``,
    but its FINAL target weight can still differ from its base weight, because
    step 3 rescales every holding including the untilted ones. What is
    preserved for those tickers is their weight RELATIVE to each other.

    ``signal_available`` should be passed explicitly (typically ``z.notna()``).
    If omitted it is inferred as ``multiplier != 1``, which is correct except
    for a genuinely observed z-score of exactly zero: that produces a
    multiplier of 1 and is indistinguishable from a missing signal by the
    multiplier alone.
    """
    if not isinstance(base_weights, pd.Series):
        raise TypeError("base_weights must be a pandas Series indexed by ticker")
    if not isinstance(multiplier, pd.Series):
        raise TypeError("multiplier must be a pandas Series indexed by ticker")
    if base_weights.index.has_duplicates:
        raise ValueError("base_weights has duplicate tickers")
    if multiplier.index.has_duplicates:
        raise ValueError("multiplier has duplicate tickers")
    if set(base_weights.index) != set(multiplier.index):
        only_base = sorted(set(base_weights.index) - set(multiplier.index))
        only_mult = sorted(set(multiplier.index) - set(base_weights.index))
        raise ValueError(
            "base_weights and multiplier cover different tickers "
            f"(base-only: {only_base[:5]}, multiplier-only: {only_mult[:5]}); "
            "a misaligned index would silently add or drop holdings"
        )

    base = base_weights.astype(float)
    mult = multiplier.reindex(base.index).astype(float)
    if not np.isfinite(base.to_numpy()).all():
        raise ValueError("base_weights contains NaN or inf")
    if (base < -WEIGHT_TOLERANCE).any():
        raise ValueError("base_weights must be non-negative")
    if not np.isfinite(mult.to_numpy()).all():
        raise ValueError("multiplier contains NaN or inf")

    if signal_available is None:
        available = mult.ne(1.0)
    else:
        available = signal_available.reindex(base.index).fillna(False).astype(bool)

    pre_normalisation = base * mult
    clipped_to_zero = pre_normalisation < 0.0
    clipped = pre_normalisation.clip(lower=0.0)
    total = float(clipped.sum())

    fallback_used = False
    if (mult == 1.0).all():
        # The tilt is inert this rebalance (no ticker had a usable signal, or
        # lam is zero), so the fund must be EXACTLY the base fund. Dividing by
        # a total that equals one only to within floating-point error would
        # nudge the weights in the last bit and make an untilted rebalance look
        # marginally different from the fund it is meant to reproduce.
        target = base.copy()
    elif not np.isfinite(total) or total <= WEIGHT_TOLERANCE:
        warnings.warn(
            "sentiment tilt clipped every holding to zero; falling back to the "
            "untilted base weights for this rebalance",
            RuntimeWarning,
            stacklevel=2,
        )
        target = base.copy()
        fallback_used = True
    else:
        target = clipped / total

    target_total = float(target.sum())
    if not np.isfinite(target_total) or abs(target_total - 1.0) > WEIGHT_TOLERANCE:
        raise ValueError(f"tilted target weights sum to {target_total:.12g}, not one")
    if (target < -WEIGHT_TOLERANCE).any():
        raise ValueError("tilted target weights are not all non-negative")

    return pd.DataFrame(
        {
            "base_weight": base,
            "tilt_multiplier": mult,
            "pre_normalisation_weight": pre_normalisation,
            "target_weight": target,
            "clipped_to_zero": clipped_to_zero,
            "signal_available": available,
            "fallback_used": fallback_used,
        },
        columns=TILT_WEIGHT_COLUMNS,
    )


def build_tilted_target_schedule(
    base_weights_by_effective_date: Mapping[pd.Timestamp, pd.Series],
    signal_date_by_effective_date: Mapping[pd.Timestamp, pd.Timestamp],
    ticker_z: pd.DataFrame,
    lam: float,
) -> tuple[dict[pd.Timestamp, pd.Series], pd.DataFrame]:
    """Tilt every rebalance's base weights and return the schedule plus detail.

    ``targets_by_effective_date`` is ready for
    ``portfolios.run_backtest_from_target_schedule``; ``detail`` carries one
    row per (effective_date, ticker) for auditing.
    """
    targets: dict[pd.Timestamp, pd.Series] = {}
    detail_frames: list[pd.DataFrame] = []

    for effective_date in sorted(base_weights_by_effective_date):
        base = base_weights_by_effective_date[effective_date].astype(float)
        base.index = pd.Index(base.index, name="ticker")
        signal_date = pd.Timestamp(signal_date_by_effective_date[effective_date])
        z = lookup_ticker_signal(ticker_z, signal_date, base.index)
        multiplier = sentiment_tilt_multiplier(z, lam)
        # signal_available comes from the z-score itself, not from
        # "multiplier != 1", so an observed z of exactly zero still counts as
        # an available signal.
        tilted = apply_sentiment_tilt(base, multiplier, signal_available=z.notna())

        targets[pd.Timestamp(effective_date)] = tilted["target_weight"]

        detail = tilted.copy()
        detail.insert(0, "effective_date", pd.Timestamp(effective_date))
        detail.insert(1, "signal_date", signal_date)
        detail.insert(2, "ticker_z", z)
        detail_frames.append(detail.reset_index())

    empty_columns = [
        "ticker", "effective_date", "signal_date", "ticker_z", *TILT_WEIGHT_COLUMNS
    ]
    detail_all = (
        pd.concat(detail_frames, ignore_index=True)
        if detail_frames
        else pd.DataFrame(columns=empty_columns)
    )
    return targets, detail_all
