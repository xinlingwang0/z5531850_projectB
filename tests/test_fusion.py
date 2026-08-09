"""Tests for the shared backtest executor and the extended sentiment tilt.

Covers the backtest extraction (the four portfolio methods and the fusion path
must be priced by one engine), the ticker-level extended z-score grid, signal
timing, the pure tilt functions, and no-look-ahead counterfactuals.

Only the extended lexicon is exercised here: the baseline-vs-extended
comparison lives in the sentiment module and its tests, which this file does
not touch.
"""
from __future__ import annotations

import pathlib
import sys
import warnings

import matplotlib

# Fix a non-interactive backend before importing anything that pulls in
# matplotlib.pyplot, so a headless run cannot abort on a GUI backend.
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import run_part_b  # noqa: E402
from src import fusion, portfolios  # noqa: E402

BASE_VARIANT = "Equity-Only Minimum Variance"
MOMENTUM_VARIANT = "Extended Sentiment Momentum"
CONTRARIAN_VARIANT = "Extended Sentiment Contrarian"
TILT_VARIANTS = [MOMENTUM_VARIANT, CONTRARIAN_VARIANT]
ALL_VARIANTS = [BASE_VARIANT, *TILT_VARIANTS]


# ============================================================================
# Shared fixtures
# ============================================================================


def _returns_panel(dates: pd.DatetimeIndex, data: dict[str, list[float]]) -> pd.DataFrame:
    panel = pd.DataFrame(data, index=dates)
    panel.index.name = "date"
    return panel


def _simple_setup():
    """Small deterministic world: 3 tickers, 40 trading days, 3 rebalances."""
    dates = pd.bdate_range("2021-01-04", periods=40)
    rng = np.random.default_rng(7)
    panel = _returns_panel(
        dates,
        {t: list(rng.normal(0.0005, 0.01, size=40)) for t in ["AAA", "BBB", "CCC"]},
    )
    effective_dates = [dates[10], dates[20], dates[30]]
    formation_by_effective = {e: dates[list(dates).index(e) - 1] for e in effective_dates}
    signal_by_effective = {
        e: fusion.previous_trading_day(e, dates) for e in effective_dates
    }
    base_weights = {}
    for effective in effective_dates:
        series = pd.Series({"AAA": 0.5, "BBB": 0.3, "CCC": 0.2})
        series.index = pd.Index(series.index, name="ticker")
        base_weights[effective] = series

    z_rows = []
    for date in dates:
        for i, ticker in enumerate(["AAA", "BBB", "CCC"]):
            # Deliberately modest z-scores: large magnitudes would drive every
            # multiplier negative and collapse the tilt into the clipping
            # fallback, which would stop these tests exercising the normal path.
            z_rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "extended_z_expanding": 0.2 * (i + 1) * (1 if date.day % 2 == 0 else -1),
                }
            )
    ticker_z = pd.DataFrame(z_rows)
    return panel, base_weights, formation_by_effective, signal_by_effective, ticker_z


# ============================================================================
# 16.1 Backtest extraction: one engine for the methods and for fusion
# ============================================================================


def test_public_backtest_entrypoint_matches_extracted_executor():
    """The original public entry point and the extracted executor must agree.

    run_walk_forward_backtest is driven through its own method path, then the
    same target schedule is handed straight to the executor; both must produce
    the same returns, weights and realised turnover/cost.
    """
    dates = pd.bdate_range("2021-01-04", periods=90)
    rng = np.random.default_rng(11)
    panel = _returns_panel(
        dates, {t: list(rng.normal(0.0004, 0.008, size=90)) for t in ["AAA", "BBB", "CCC"]}
    )

    via_public = portfolios.run_walk_forward_backtest(
        panel, method="equal_weight", window=20, annualization_factor=252
    )
    targets, _rows, diagnostic_rows = portfolios._build_target_schedule(
        panel, "equal_weight", 20
    )
    formation_by_effective = {
        pd.Timestamp(r["effective_date"]): pd.Timestamp(r["formation_date"])
        for r in diagnostic_rows
    }
    via_executor = portfolios.run_backtest_from_target_schedule(
        panel, targets, formation_by_effective
    )

    pd.testing.assert_frame_equal(via_public.returns, via_executor.returns)
    pd.testing.assert_frame_equal(via_public.weights, via_executor.weights)

    public_costs = via_public.diagnostics.dropna(subset=["turnover"]).set_index("effective_date")
    exec_costs = via_executor.diagnostics.set_index("effective_date")
    np.testing.assert_allclose(
        public_costs["turnover"].to_numpy(dtype=float),
        exec_costs["turnover"].to_numpy(dtype=float),
    )
    np.testing.assert_allclose(
        public_costs["transaction_cost"].to_numpy(dtype=float),
        exec_costs["transaction_cost"].to_numpy(dtype=float),
    )


def test_executor_applies_target_on_its_effective_date_not_its_formation_date():
    dates = pd.bdate_range("2021-01-04", periods=5)
    panel = _returns_panel(
        dates, {"AAA": [0.10, 0.0, 0.0, 0.0, 0.0], "BBB": [0.0, 0.0, 0.0, 0.0, 0.0]}
    )
    effective = dates[2]
    targets = {effective: pd.Series({"AAA": 1.0, "BBB": 0.0})}
    result = portfolios.run_backtest_from_target_schedule(
        panel, targets, {effective: dates[0]}
    )
    # The book only exists from the effective date onwards, never from the
    # formation date, so nothing is priced before it.
    assert result.returns.index.min() == effective
    assert len(result.returns) == 3


def test_executor_ignores_the_formation_date_value_when_pricing():
    """Changing only the audit formation_date must not move any return."""
    panel, base_weights, formation_by_effective, _sig, _z = _simple_setup()
    targets = dict(base_weights)

    first = portfolios.run_backtest_from_target_schedule(
        panel, targets, formation_by_effective
    )
    shifted = {e: panel.index[0] for e in formation_by_effective}
    second = portfolios.run_backtest_from_target_schedule(panel, targets, shifted)

    pd.testing.assert_frame_equal(first.returns, second.returns)
    np.testing.assert_allclose(
        first.weights["target_weight"].to_numpy(), second.weights["target_weight"].to_numpy()
    )
    # ...but it is still recorded, so it remains auditable.
    assert set(second.weights["formation_date"]) == {panel.index[0]}


def test_executor_rejects_formation_date_on_or_after_effective_date():
    panel, base_weights, _f, _s, _z = _simple_setup()
    effective = min(base_weights)
    with pytest.raises(ValueError, match="strictly before"):
        portfolios.run_backtest_from_target_schedule(
            panel, {effective: base_weights[effective]}, {effective: effective}
        )


def test_executor_weight_drift_turnover_and_returns_on_a_hand_checked_schedule():
    dates = pd.bdate_range("2021-01-04", periods=3)
    panel = _returns_panel(dates, {"AAA": [0.10, 0.00, 0.0], "BBB": [0.00, 0.20, 0.0]})
    effective = dates[0]
    targets = {effective: pd.Series({"AAA": 0.5, "BBB": 0.5})}
    result = portfolios.run_backtest_from_target_schedule(
        panel, targets, {effective: pd.Timestamp("2020-12-31")},
        transaction_cost_rate=0.001,
    )

    # Day 0: build from cash -> turnover 1.0, cost 0.001.
    assert result.diagnostics.loc[0, "turnover"] == pytest.approx(1.0)
    assert result.diagnostics.loc[0, "transaction_cost"] == pytest.approx(0.001)
    assert result.returns["gross_return"].iloc[0] == pytest.approx(0.5 * 0.10)
    assert result.returns["net_return"].iloc[0] == pytest.approx(0.05 - 0.001)

    # Day 1: buy-and-hold drift -> AAA 0.5*1.1 = 0.55, BBB 0.5, renormalised.
    drifted_aaa = 0.55 / 1.05
    drifted_bbb = 0.50 / 1.05
    assert drifted_aaa + drifted_bbb == pytest.approx(1.0)
    assert result.returns["gross_return"].iloc[1] == pytest.approx(drifted_bbb * 0.20)
    # No rebalance on day 1, so no further cost.
    assert result.returns["transaction_cost"].iloc[1] == pytest.approx(0.0)


def test_executor_rejects_illegal_target_schedules():
    panel, base_weights, formation, _s, _z = _simple_setup()
    effective = min(base_weights)
    with pytest.raises(ValueError, match="sums to"):
        portfolios.run_backtest_from_target_schedule(
            panel, {effective: pd.Series({"AAA": 0.3, "BBB": 0.3, "CCC": 0.3})},
            {effective: formation[effective]},
        )
    with pytest.raises(ValueError, match="long-only"):
        portfolios.run_backtest_from_target_schedule(
            panel, {effective: pd.Series({"AAA": 1.4, "BBB": -0.2, "CCC": -0.2})},
            {effective: formation[effective]},
        )
    with pytest.raises(ValueError, match="absent from"):
        portfolios.run_backtest_from_target_schedule(
            panel, {effective: pd.Series({"AAA": 0.5, "ZZZ": 0.5})},
            {effective: formation[effective]},
        )


# ============================================================================
# 16.3 Signal date and lag
# ============================================================================


def test_previous_trading_day_is_strictly_before_and_skips_the_weekend():
    calendar = pd.bdate_range("2021-01-04", periods=10)
    monday = pd.Timestamp("2021-01-11")
    assert monday in set(calendar)
    assert fusion.previous_trading_day(monday, calendar) == pd.Timestamp("2021-01-08")
    for effective in calendar[1:]:
        assert fusion.previous_trading_day(effective, calendar) < effective


def test_previous_trading_day_raises_when_no_earlier_trading_day_exists():
    calendar = pd.bdate_range("2021-01-04", periods=5)
    with pytest.raises(ValueError, match="no trading day strictly before"):
        fusion.previous_trading_day(calendar[0], calendar)


def test_signal_lookup_is_exact_and_never_reaches_backwards_or_forwards():
    ticker_z = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-04", "2021-01-06", "2021-01-07"]),
            "ticker": ["AAA", "AAA", "AAA"],
            "extended_z_expanding": [1.0, np.nan, 3.0],
        }
    )
    tickers = ["AAA"]

    # Exact hit.
    assert fusion.lookup_ticker_signal(
        ticker_z, pd.Timestamp("2021-01-04"), tickers
    ).iloc[0] == pytest.approx(1.0)

    # Row absent for that date -> NaN, NOT the most recent earlier value.
    assert pd.isna(
        fusion.lookup_ticker_signal(ticker_z, pd.Timestamp("2021-01-05"), tickers).iloc[0]
    )

    # Row present but z missing -> NaN, not carried forward from 2021-01-04.
    assert pd.isna(
        fusion.lookup_ticker_signal(ticker_z, pd.Timestamp("2021-01-06"), tickers).iloc[0]
    )

    # A ticker absent from the table entirely -> NaN, never a future value.
    assert pd.isna(
        fusion.lookup_ticker_signal(ticker_z, pd.Timestamp("2021-01-04"), ["ZZZ"]).iloc[0]
    )


def test_signal_only_available_from_the_effective_date_onwards_returns_nan():
    """A ticker whose first z-score is on/after the effective date is unusable."""
    calendar = pd.bdate_range("2021-01-04", periods=6)
    effective = calendar[3]
    signal_date = fusion.previous_trading_day(effective, calendar)
    ticker_z = pd.DataFrame(
        {
            "date": [effective, calendar[4]],
            "ticker": ["AAA", "AAA"],
            "extended_z_expanding": [2.0, 2.0],
        }
    )
    assert pd.isna(fusion.lookup_ticker_signal(ticker_z, signal_date, ["AAA"]).iloc[0])


def test_monday_signal_can_only_affect_a_later_effective_date():
    calendar = pd.bdate_range("2021-01-04", periods=10)
    monday = pd.Timestamp("2021-01-11")
    tuesday = pd.Timestamp("2021-01-12")
    assert monday.dayofweek == 0 and tuesday.dayofweek == 1

    # Monday's own effective date uses Friday, not Monday.
    assert fusion.previous_trading_day(monday, calendar) == pd.Timestamp("2021-01-08")
    # Monday's signal is first usable by Tuesday's target.
    assert fusion.previous_trading_day(tuesday, calendar) == monday


# ============================================================================
# 16.4 Pure tilt functions
# ============================================================================


def _base() -> pd.Series:
    return pd.Series({"AAA": 0.5, "BBB": 0.3, "CCC": 0.2})


def test_lambda_zero_leaves_every_weight_untouched():
    z = pd.Series({"AAA": 2.0, "BBB": -2.0, "CCC": np.nan})
    multiplier = fusion.sentiment_tilt_multiplier(z, 0.0)
    assert (multiplier == 1.0).all()
    result = fusion.apply_sentiment_tilt(_base(), multiplier, signal_available=z.notna())
    np.testing.assert_array_equal(result["target_weight"].to_numpy(), _base().to_numpy())


def test_positive_z_raises_and_negative_z_lowers_the_multiplier_when_lambda_positive():
    z = pd.Series({"AAA": 1.5, "BBB": -1.5, "CCC": 0.0})
    multiplier = fusion.sentiment_tilt_multiplier(z, 1.0)
    assert multiplier["AAA"] > 1.0
    assert multiplier["BBB"] < 1.0
    assert multiplier["CCC"] == pytest.approx(1.0)


def test_negative_lambda_reverses_the_direction_for_the_same_z():
    z = pd.Series({"AAA": 1.5, "BBB": -1.5, "CCC": 0.0})
    momentum = fusion.sentiment_tilt_multiplier(z, 1.0)
    contrarian = fusion.sentiment_tilt_multiplier(z, -1.0)
    assert (momentum["AAA"] - 1.0) * (contrarian["AAA"] - 1.0) < 0
    assert (momentum["BBB"] - 1.0) * (contrarian["BBB"] - 1.0) < 0


def test_missing_z_gives_multiplier_one_and_untouched_pre_normalisation_weight():
    z = pd.Series({"AAA": 2.0, "BBB": np.nan, "CCC": np.nan})
    multiplier = fusion.sentiment_tilt_multiplier(z, 1.0)
    assert multiplier["BBB"] == 1.0 and multiplier["CCC"] == 1.0
    result = fusion.apply_sentiment_tilt(_base(), multiplier, signal_available=z.notna())
    assert result.loc["BBB", "pre_normalisation_weight"] == pytest.approx(_base()["BBB"])
    assert result.loc["CCC", "pre_normalisation_weight"] == pytest.approx(_base()["CCC"])
    assert not result.loc["BBB", "signal_available"]
    assert result.loc["AAA", "signal_available"]


def test_tickers_without_a_signal_keep_their_relative_proportions():
    z = pd.Series({"AAA": 2.0, "BBB": np.nan, "CCC": np.nan})
    multiplier = fusion.sentiment_tilt_multiplier(z, 1.0)
    result = fusion.apply_sentiment_tilt(_base(), multiplier, signal_available=z.notna())
    base = _base()
    assert result.loc["BBB", "target_weight"] / result.loc["CCC", "target_weight"] == pytest.approx(
        base["BBB"] / base["CCC"]
    )


def test_all_signals_missing_reproduces_the_base_weights_exactly():
    z = pd.Series({"AAA": np.nan, "BBB": np.nan, "CCC": np.nan})
    multiplier = fusion.sentiment_tilt_multiplier(z, 1.0)
    result = fusion.apply_sentiment_tilt(_base(), multiplier, signal_available=z.notna())
    # Exact, not merely close: an inert tilt must not perturb the fund at all.
    np.testing.assert_array_equal(result["target_weight"].to_numpy(), _base().to_numpy())
    assert not result["signal_available"].any()


def test_clipping_records_negative_pre_normalisation_weights():
    z = pd.Series({"AAA": 2.0, "BBB": 0.0, "CCC": 0.0})
    multiplier = fusion.sentiment_tilt_multiplier(z, -1.0)  # AAA -> 1 - 2 = -1
    result = fusion.apply_sentiment_tilt(_base(), multiplier, signal_available=z.notna())
    assert result.loc["AAA", "pre_normalisation_weight"] < 0
    assert result.loc["AAA", "clipped_to_zero"]
    assert result.loc["AAA", "target_weight"] == pytest.approx(0.0)
    assert not result.loc["BBB", "clipped_to_zero"]
    assert result["target_weight"].sum() == pytest.approx(1.0)
    assert (result["target_weight"] >= 0).all()


def test_all_weights_clipped_falls_back_to_base_and_flags_it():
    base = pd.Series({"AAA": 0.5, "BBB": 0.5})
    z = pd.Series({"AAA": 2.0, "BBB": 2.0})
    multiplier = fusion.sentiment_tilt_multiplier(z, -1.0)  # both -> -1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fusion.apply_sentiment_tilt(base, multiplier, signal_available=z.notna())
    assert any("clipped every holding" in str(w.message) for w in caught)
    assert result["fallback_used"].all()
    np.testing.assert_allclose(result["target_weight"].to_numpy(), base.to_numpy())
    assert result["target_weight"].sum() == pytest.approx(1.0)
    assert (result["target_weight"] >= 0).all()


def test_misaligned_indexes_raise_instead_of_silently_adding_or_dropping_tickers():
    with pytest.raises(ValueError, match="different tickers"):
        fusion.apply_sentiment_tilt(pd.Series({"AAA": 1.0}), pd.Series({"BBB": 1.0}))


def test_duplicate_tickers_are_rejected():
    duplicated = pd.Index(["AAA", "AAA"])
    with pytest.raises(ValueError, match="duplicate"):
        fusion.apply_sentiment_tilt(
            pd.Series([0.5, 0.5], index=duplicated),
            pd.Series([1.0, 1.0], index=duplicated),
        )


def test_negative_base_weights_are_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        fusion.apply_sentiment_tilt(
            pd.Series({"AAA": 1.2, "BBB": -0.2}), pd.Series({"AAA": 1.0, "BBB": 1.0})
        )


# ============================================================================
# 16.6 No-look-ahead counterfactuals
# ============================================================================


def _build_schedule(ticker_z: pd.DataFrame, setup, lam: float = 1.0):
    _panel, base_weights, _formation, signal_by_effective, _z = setup
    return fusion.build_tilted_target_schedule(
        base_weights, signal_by_effective, ticker_z, lam
    )


def test_counterfactual_a_future_signal_cannot_change_earlier_target_weights():
    setup = _simple_setup()
    _panel, base_weights, _formation, signal_by_effective, ticker_z = setup
    original_targets, _detail = _build_schedule(ticker_z, setup)

    effective_dates = sorted(base_weights)
    cutoff_signal_date = signal_by_effective[effective_dates[1]]

    mutated = ticker_z.copy()
    mutated.loc[mutated["date"] >= cutoff_signal_date, "extended_z_expanding"] = 0.9
    mutated_targets, _detail2 = _build_schedule(mutated, setup)

    changed_any = False
    for effective_date in effective_dates:
        if signal_by_effective[effective_date] < cutoff_signal_date:
            np.testing.assert_array_equal(
                original_targets[effective_date].to_numpy(),
                mutated_targets[effective_date].to_numpy(),
            )
        elif not np.allclose(
            original_targets[effective_date].to_numpy(),
            mutated_targets[effective_date].to_numpy(),
        ):
            changed_any = True
    # The mutation must actually bite somewhere, otherwise the test proves nothing.
    assert changed_any


def test_counterfactual_b_signal_date_change_cannot_move_that_day_or_earlier_returns():
    setup = _simple_setup()
    panel, base_weights, formation_by_effective, signal_by_effective, ticker_z = setup
    effective_dates = sorted(base_weights)
    target_effective = effective_dates[1]
    signal_date = signal_by_effective[target_effective]

    def run(z_table):
        targets, _detail = fusion.build_tilted_target_schedule(
            base_weights, signal_by_effective, z_table, 1.0
        )
        result = portfolios.run_backtest_from_target_schedule(
            panel, targets, formation_by_effective
        )
        return targets, result.returns

    original_targets, original_returns = run(ticker_z)

    # Mutate one ticker only, by a moderate amount, so the normal tilt path is
    # exercised rather than the all-clipped fallback.
    mutated = ticker_z.copy()
    on_signal_day = (mutated["date"] == signal_date) & (mutated["ticker"] == "AAA")
    assert on_signal_day.any()
    mutated.loc[on_signal_day, "extended_z_expanding"] = 0.75
    mutated_targets, mutated_returns = run(mutated)

    # Returns strictly before the signal date are untouched.
    before = original_returns.index < signal_date
    pd.testing.assert_frame_equal(original_returns.loc[before], mutated_returns.loc[before])

    # The signal date's OWN realised return is untouched: the new target is
    # only held from the next effective date onwards.
    assert signal_date in original_returns.index
    pd.testing.assert_series_equal(
        original_returns.loc[signal_date], mutated_returns.loc[signal_date]
    )

    # The target that uses this signal date is allowed to change, and does.
    assert not np.allclose(
        original_targets[target_effective].to_numpy(),
        mutated_targets[target_effective].to_numpy(),
    )


def test_counterfactual_c_past_weights_and_past_returns_are_asserted_separately():
    setup = _simple_setup()
    panel, base_weights, formation_by_effective, signal_by_effective, ticker_z = setup
    effective_dates = sorted(base_weights)
    cutoff_effective = effective_dates[2]
    cutoff_signal = signal_by_effective[cutoff_effective]

    def run(z_table):
        targets, _detail = fusion.build_tilted_target_schedule(
            base_weights, signal_by_effective, z_table, 1.0
        )
        result = portfolios.run_backtest_from_target_schedule(
            panel, targets, formation_by_effective
        )
        return targets, result.returns

    original_targets, original_returns = run(ticker_z)
    mutated = ticker_z.copy()
    mutated.loc[mutated["date"] >= cutoff_signal, "extended_z_expanding"] = 0.9
    mutated_targets, mutated_returns = run(mutated)

    # (1) past WEIGHTS unchanged - asserted on its own
    for effective_date in effective_dates:
        if signal_by_effective[effective_date] < cutoff_signal:
            np.testing.assert_array_equal(
                original_targets[effective_date].to_numpy(),
                mutated_targets[effective_date].to_numpy(),
            )

    # (2) past RETURNS unchanged - asserted separately, up to and including
    #     the cutoff signal date itself
    upto = original_returns.index <= cutoff_signal
    pd.testing.assert_frame_equal(original_returns.loc[upto], mutated_returns.loc[upto])


# ============================================================================
# Real-output checks (skipped when the build has not been run)
# ============================================================================


def _load(path: pathlib.Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        pytest.skip(f"{path.name} not built yet")
    return pd.read_csv(path, **kwargs)


def test_ticker_extended_z_grid_is_complete_and_unfilled():
    tz = _load(ROOT / "results/data/ticker_sentiment_z.csv", parse_dates=["date"])
    assert list(tz.columns) == fusion.TICKER_Z_COLUMNS
    assert "baseline_z_expanding" not in tz.columns
    assert tz["date"].min() >= pd.Timestamp("2021-01-01")
    assert tz["date"].max() <= pd.Timestamp("2023-12-31")
    assert not (tz["date"].dt.dayofweek >= 5).any()
    assert tz["ticker"].nunique() == 50
    assert not tz.duplicated(["date", "ticker"]).any()
    assert len(tz) == tz["date"].nunique() * 50
    # Missing (date, ticker) combinations stay missing.
    assert tz["extended_z_expanding"].isna().any()
    assert not (tz["extended_z_expanding"].fillna(0) == 0).all()


def test_fusion_outputs_exist_with_the_required_schema():
    summary = _load(ROOT / "results/tables/fusion_before_vs_after.csv")
    assert len(summary) == 3
    for column in [
        "variant", "lexicon", "lambda", "direction", "annual_return",
        "annual_volatility", "sharpe_ratio", "max_drawdown",
        "cumulative_gross_return", "cumulative_net_return", "initial_turnover",
        "avg_rebalance_turnover_ex_initial", "total_turnover",
    ]:
        assert column in summary.columns, column
    assert set(summary["variant"]) == set(ALL_VARIANTS)

    returns = _load(ROOT / "results/data/fusion_returns.csv", parse_dates=["date"])
    assert set(returns["variant"]) == set(ALL_VARIANTS)
    assert returns["date"].min() >= pd.Timestamp("2021-01-01")
    assert returns["date"].max() <= pd.Timestamp("2023-12-31")

    weights = _load(
        ROOT / "results/data/fusion_weights.csv",
        parse_dates=["formation_date", "effective_date", "signal_date"],
    )
    assert set(weights["variant"]) == set(ALL_VARIANTS)
    for column in [
        "signal_date", "sector", "base_weight", "ticker_z", "tilt_multiplier",
        "pre_normalisation_weight", "target_weight", "signal_available",
        "clipped_to_zero", "fallback_used",
    ]:
        assert column in weights.columns, column
    # Weights/coverage may keep the 2020-12-31 audit record, but never 2024.
    assert not (weights["effective_date"].dt.year == 2024).any()
    assert not (weights["formation_date"].dt.year == 2024).any()

    coverage = _load(
        ROOT / "results/tables/ticker_sentiment_coverage_by_formation_date.csv",
        parse_dates=["formation_date", "effective_date", "signal_date"],
    )
    # Extended only - no baseline rows at the fusion stage.
    assert set(coverage["lexicon"]) == {"extended"}
    assert not coverage["formation_date"].duplicated().any()
    assert (coverage["signal_date"] < coverage["effective_date"]).all()

    diagnostics = _load(ROOT / "results/tables/fusion_diagnostics.csv")
    assert len(diagnostics) == 3
    assert (ROOT / "results/figures/figure_8_fusion_before_vs_after.png").exists()


def test_real_tilt_weights_are_legal_at_every_rebalance():
    weights = _load(
        ROOT / "results/data/fusion_weights.csv", parse_dates=["effective_date"]
    )
    tilts = weights.loc[weights["variant"] != BASE_VARIANT]
    assert set(tilts["variant"]) == set(TILT_VARIANTS)
    for (variant, effective_date), group in tilts.groupby(["variant", "effective_date"]):
        values = group["target_weight"].to_numpy(dtype=float)
        assert np.isfinite(values).all(), (variant, effective_date)
        assert (values >= -1e-9).all(), (variant, effective_date)
        assert values.sum() == pytest.approx(1.0, abs=1e-9), (variant, effective_date)
        assert not group["ticker"].duplicated().any(), (variant, effective_date)
        assert len(group) == 50, (variant, effective_date)


def test_real_base_variant_reuses_the_existing_fund_rather_than_recomputing_it():
    fusion_returns = _load(ROOT / "results/data/fusion_returns.csv", parse_dates=["date"])
    fund_returns = _load(ROOT / "results/data/fund_returns.csv", parse_dates=["date"])

    base = fusion_returns.loc[fusion_returns["variant"] == BASE_VARIANT].sort_values("date")
    original = fund_returns.loc[
        fund_returns["fund_name"] == "Equity-Only Minimum Variance"
    ].sort_values("date")
    assert len(base) == len(original)
    for column in ["gross_return", "transaction_cost", "net_return", "growth_of_1", "drawdown"]:
        np.testing.assert_array_equal(base[column].to_numpy(), original[column].to_numpy())
    assert (base["lexicon"] == "none").all()
    assert (base["lambda"] == 0.0).all()
    assert (base["direction"] == "base").all()

    fusion_weights = _load(
        ROOT / "results/data/fusion_weights.csv", parse_dates=["effective_date"]
    )
    fund_weights = _load(
        ROOT / "results/data/fund_weights.csv", parse_dates=["effective_date"]
    )
    base_w = fusion_weights.loc[fusion_weights["variant"] == BASE_VARIANT].sort_values(
        ["effective_date", "ticker"]
    )
    original_w = fund_weights.loc[
        (fund_weights["fund_family"] == "Equity-Only")
        & (fund_weights["method"] == "minimum_variance")
    ].sort_values(["effective_date", "ticker"])
    np.testing.assert_array_equal(
        base_w["target_weight"].to_numpy(), original_w["target_weight"].to_numpy()
    )


def test_real_first_formation_date_applies_no_tilt_and_borrows_no_signal():
    weights = _load(
        ROOT / "results/data/fusion_weights.csv",
        parse_dates=["formation_date", "effective_date", "signal_date"],
    )
    first = weights.loc[weights["formation_date"] == pd.Timestamp("2020-12-31")]
    if first.empty:
        pytest.skip("no 2020-12-31 formation date in the built outputs")
    assert set(first["effective_date"].unique()) == {pd.Timestamp("2021-01-04")}
    assert set(first["signal_date"].unique()) == {pd.Timestamp("2020-12-31")}

    tilts = first.loc[first["variant"] != BASE_VARIANT]
    assert tilts["ticker_z"].isna().all()
    assert not tilts["signal_available"].any()
    assert (tilts["tilt_multiplier"] == 1.0).all()

    base = first.loc[first["variant"] == BASE_VARIANT].set_index("ticker")["target_weight"]
    for _variant, group in tilts.groupby("variant"):
        aligned = group.set_index("ticker")["target_weight"].reindex(base.index)
        np.testing.assert_array_equal(aligned.to_numpy(), base.to_numpy())

    coverage = _load(
        ROOT / "results/tables/ticker_sentiment_coverage_by_formation_date.csv",
        parse_dates=["formation_date"],
    )
    first_cov = coverage.loc[coverage["formation_date"] == pd.Timestamp("2020-12-31")]
    assert (first_cov["coverage_ratio"] == 0).all()
    assert (first_cov["base_weight_with_z"] == 0).all()
    assert not first_cov["tilt_active"].any()

    # And no 2020 ticker z-score exists to have been borrowed from.
    tz = _load(ROOT / "results/data/ticker_sentiment_z.csv", parse_dates=["date"])
    assert not (tz["date"].dt.year == 2020).any()


def test_figure_8_plots_exactly_three_strategy_curves_over_2021_2023(monkeypatch, tmp_path):
    """Figure 8 must show the base plus the two extended tilts - no baseline
    fusion curve - and must plot only 2021-2023 data.
    """
    import matplotlib.pyplot as plt

    returns = _load(ROOT / "results/data/fusion_returns.csv", parse_dates=["date"])

    monkeypatch.setattr(run_part_b, "FIGURES", tmp_path)
    captured = {}
    original_close = plt.close

    def _capture_instead_of_close(*args, **kwargs):
        captured["fig"] = plt.gcf()

    monkeypatch.setattr(plt, "close", _capture_instead_of_close)
    try:
        run_part_b.apply_theme()
        run_part_b.plot_fusion_before_vs_after(returns)
    finally:
        monkeypatch.setattr(plt, "close", original_close)

    fig = captured["fig"]
    ax = fig.axes[0]
    labelled = [
        line for line in ax.get_lines() if not line.get_label().startswith("_")
    ]
    labels = sorted(line.get_label() for line in labelled)
    assert labels == sorted(ALL_VARIANTS), labels
    assert not any("Baseline" in label for label in labels)

    for line in labelled:
        xdata = pd.to_datetime(line.get_xdata())
        assert xdata.min() >= pd.Timestamp("2021-01-01")
        assert xdata.max() <= pd.Timestamp("2023-12-31")
    original_close(fig)


def test_figure_8_rejects_out_of_period_input():
    returns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-12-31", "2021-01-04"]),
            "variant": BASE_VARIANT,
            "growth_of_1": [1.0, 1.01],
        }
    )
    with pytest.raises(AssertionError, match="2020"):
        run_part_b.plot_fusion_before_vs_after(returns)


def test_real_cumulative_returns_are_compounded_not_summed():
    returns = _load(ROOT / "results/data/fusion_returns.csv", parse_dates=["date"])
    summary = _load(ROOT / "results/tables/fusion_before_vs_after.csv")
    for variant, group in returns.groupby("variant"):
        group = group.sort_values("date")
        expected_net = (1 + group["net_return"]).prod() - 1
        expected_gross = (1 + group["gross_return"]).prod() - 1
        row = summary.loc[summary["variant"] == variant].iloc[0]
        assert row["cumulative_net_return"] == pytest.approx(expected_net, rel=1e-10)
        assert row["cumulative_gross_return"] == pytest.approx(expected_gross, rel=1e-10)
        # growth_of_1 is built from net returns and must agree with them.
        assert group["growth_of_1"].iloc[-1] == pytest.approx(1 + expected_net, rel=1e-10)
        # ...and must not equal the (wrong) simple sum of daily returns.
        assert row["cumulative_net_return"] != pytest.approx(group["net_return"].sum())
