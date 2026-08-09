"""Tests for the RiskBridge Funds Streamlit app.

Three layers:

* the CSV loaders and their schema validation, against the real results files;
* the pure calculations (buy-and-hold allocation, rolling display smoother,
  dynamic copy), against hand-checked fixtures;
* the app itself through ``streamlit.testing.v1.AppTest``, exercising every
  page and the interactions on them.

Required results files must exist: a missing one is a real failure, not a
reason to skip.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = pathlib.Path(__file__).resolve().parent.parent
APP_PATH = ROOT / "streamlit_app.py"
sys.path.insert(0, str(ROOT))


def _load_app_module():
    """Import streamlit_app as a module without executing its UI.

    ``main()`` is guarded by ``if __name__ == "__main__"``, so importing under
    any other name gives access to the pure functions on their own.
    """
    spec = importlib.util.spec_from_file_location("riskbridge_app", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


app = _load_app_module()

REQUIRED_CSVS = [
    "results/data/fund_returns.csv",
    "results/data/fusion_returns.csv",
    "results/data/sector_sentiment_index.csv",
    "results/tables/performance_metrics.csv",
    "results/tables/current_holdings.csv",
    "results/tables/fusion_before_vs_after.csv",
    "results/tables/ticker_sentiment_coverage_by_formation_date.csv",
]

DEFAULT_ALLOCATION_FUNDS = [
    "Equity-Only Minimum Variance",
    "Crypto-Only HRP",
    "Combined Maximum Sharpe",
]

APP_TIMEOUT = 60


def run_app() -> AppTest:
    at = AppTest.from_file(str(APP_PATH), default_timeout=APP_TIMEOUT)
    return at.run()


# ============================================================================
# 16.1 Data loaders and schema validation
# ============================================================================


def test_every_required_results_file_exists():
    missing = [name for name in REQUIRED_CSVS if not (ROOT / name).exists()]
    assert not missing, f"missing required results file(s): {missing}"


def test_loaders_return_validated_frames_with_parsed_dates():
    returns = app.load_fund_returns()
    assert pd.api.types.is_datetime64_any_dtype(returns["date"])
    assert not returns.duplicated(["date", "fund_name"]).any()

    metrics = app.load_performance_metrics()
    assert pd.api.types.is_datetime64_any_dtype(metrics["oos_start"])
    assert metrics["fund_name"].is_unique

    sentiment = app.load_sector_sentiment()
    assert pd.api.types.is_datetime64_any_dtype(sentiment["date"])
    assert not sentiment.duplicated(["date", "sector"]).any()

    fusion_returns = app.load_fusion_returns()
    assert not fusion_returns.duplicated(["date", "variant"]).any()

    assert len(app.load_fusion_summary()) == 3
    assert not app.load_fusion_coverage().empty
    assert not app.load_current_holdings().empty


def test_missing_file_raises_a_readable_app_data_error(tmp_path):
    with pytest.raises(app.AppDataError, match="missing"):
        app._read_results_csv(tmp_path / "absent.csv", ["date"])


def test_missing_column_raises_a_readable_app_data_error(tmp_path):
    path = tmp_path / "partial.csv"
    pd.DataFrame({"date": ["2021-01-04"]}).to_csv(path, index=False)
    with pytest.raises(app.AppDataError, match="missing expected column"):
        app._read_results_csv(path, ["date", "growth_of_1"], date_columns=["date"])


def test_empty_file_raises_a_readable_app_data_error(tmp_path):
    path = tmp_path / "empty.csv"
    pd.DataFrame({"date": [], "value": []}).to_csv(path, index=False)
    with pytest.raises(app.AppDataError, match="no rows"):
        app._read_results_csv(path, ["date", "value"])


# ============================================================================
# 16.2 Allocation pure function
# ============================================================================


def _two_fund_fixture() -> pd.DataFrame:
    """Fund A doubles over three days; fund B is flat. Both start on day 1."""
    dates = pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-06"])
    rows = []
    for date, nav in zip(dates, [1.0, 1.5, 2.0]):
        rows.append({"date": date, "fund_name": "A", "growth_of_1": nav})
    for date, nav in zip(dates, [1.0, 1.0, 1.0]):
        rows.append({"date": date, "fund_name": "B", "growth_of_1": nav})
    return pd.DataFrame(rows)


def test_allocation_normalises_each_fund_to_one_and_applies_initial_weights():
    result = app.calculate_buy_and_hold_allocation(
        _two_fund_fixture(), {"A": 0.5, "B": 0.5}, 1000.0
    )
    timeline = result["timeline"]
    # Day 1: 0.5*1 + 0.5*1 = 1.0
    assert timeline["portfolio_nav"].iloc[0] == pytest.approx(1.0)
    assert timeline["portfolio_value"].iloc[0] == pytest.approx(1000.0)
    # Day 3: 0.5*2 + 0.5*1 = 1.5
    assert timeline["portfolio_nav"].iloc[-1] == pytest.approx(1.5)
    assert result["ending_value"] == pytest.approx(1500.0)
    assert result["cumulative_return"] == pytest.approx(0.5)


def test_allocation_lets_the_outperformer_drift_instead_of_rebalancing_daily():
    """Buy-and-hold must differ from compounding a fixed-weight daily return.

    With a 50/50 start, fund A doubling and fund B flat, buy-and-hold ends at
    1.5. Rebalancing to 50/50 every day would compound the average daily
    return instead and give a different answer - if these two ever agree, the
    implementation has silently become a daily-rebalanced product.
    """
    frame = _two_fund_fixture()
    buy_and_hold = app.calculate_buy_and_hold_allocation(
        frame, {"A": 0.5, "B": 0.5}, 1.0
    )["timeline"]["portfolio_nav"].iloc[-1]

    nav = frame.pivot(index="date", columns="fund_name", values="growth_of_1")
    daily_returns = nav.pct_change().fillna(0.0)
    rebalanced = float((1 + (0.5 * daily_returns["A"] + 0.5 * daily_returns["B"])).prod())

    assert buy_and_hold == pytest.approx(1.5)
    assert rebalanced != pytest.approx(buy_and_hold)
    # A's implied share has drifted above its 50% start.
    assert (0.5 * 2.0) / 1.5 > 0.5


def test_allocation_uses_the_union_calendar_and_holds_nav_flat_on_non_trading_days():
    rows = [
        # Fund X trades every calendar day including the weekend.
        {"date": "2021-01-01", "fund_name": "X", "growth_of_1": 1.0},
        {"date": "2021-01-02", "fund_name": "X", "growth_of_1": 1.1},
        {"date": "2021-01-03", "fund_name": "X", "growth_of_1": 1.2},
        {"date": "2021-01-04", "fund_name": "X", "growth_of_1": 1.3},
        # Fund Y only starts on 2021-01-04.
        {"date": "2021-01-04", "fund_name": "Y", "growth_of_1": 1.0},
        {"date": "2021-01-05", "fund_name": "Y", "growth_of_1": 1.5},
    ]
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"])

    result = app.calculate_buy_and_hold_allocation(frame, {"X": 0.5, "Y": 0.5}, 100.0)
    timeline = result["timeline"].set_index("date")

    # Common start is Y's first day, not X's - nothing is filled backwards.
    assert result["common_start"] == pd.Timestamp("2021-01-04")
    assert pd.Timestamp("2021-01-01") not in timeline.index
    # Union calendar extends to Y's last day.
    assert result["end_date"] == pd.Timestamp("2021-01-05")
    # On 2021-01-05, X does not trade, so its NAV is held flat at its 01-04
    # value (normalised to 1): 0.5*1.0 + 0.5*1.5 = 1.25.
    assert timeline["portfolio_nav"].iloc[-1] == pytest.approx(1.25)
    assert not timeline["portfolio_nav"].isna().any()


def test_allocation_maximum_drawdown_is_computed_from_the_portfolio_nav():
    dates = pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-06"])
    rows = []
    for date, nav in zip(dates, [1.0, 2.0, 1.0]):
        rows.append({"date": date, "fund_name": "A", "growth_of_1": nav})
    for date, nav in zip(dates, [1.0, 1.0, 1.0]):
        rows.append({"date": date, "fund_name": "B", "growth_of_1": nav})
    frame = pd.DataFrame(rows)

    result = app.calculate_buy_and_hold_allocation(frame, {"A": 1.0, "B": 0.0}, 1.0)
    # NAV path 1.0 -> 2.0 -> 1.0, so the trough is 50% below the peak.
    assert result["max_drawdown"] == pytest.approx(-0.5)


@pytest.mark.parametrize(
    "allocations, message",
    [
        ({"A": 0.5, "B": 0.4}, "add up to 100"),
        ({"A": 1.2, "B": -0.2}, "negative"),
        ({"A": 1.0}, "between 2 and 4"),
    ],
)
def test_allocation_rejects_invalid_configurations(allocations, message):
    with pytest.raises(ValueError, match=message):
        app.calculate_buy_and_hold_allocation(_two_fund_fixture(), allocations, 1000.0)


def test_allocation_rejects_more_than_four_funds():
    dates = pd.to_datetime(["2021-01-04", "2021-01-05"])
    rows = [
        {"date": date, "fund_name": name, "growth_of_1": 1.0}
        for name in list("ABCDE")
        for date in dates
    ]
    with pytest.raises(ValueError, match="between 2 and 4"):
        app.calculate_buy_and_hold_allocation(
            pd.DataFrame(rows), dict.fromkeys("ABCDE", 0.2), 1000.0
        )


@pytest.mark.parametrize("n_funds", [2, 3, 4])
def test_default_allocation_shares_total_exactly_one_hundred(n_funds):
    """Three-way splits must not open the page on a 99.99% error."""
    shares = app.equal_default_allocation_pct(n_funds)
    assert len(shares) == n_funds
    assert sum(shares) == pytest.approx(100.0, abs=1e-9)
    assert all(share >= 0 for share in shares)
    # And the validator accepts them without any nudging.
    frame = pd.DataFrame(
        [
            {"date": pd.Timestamp("2021-01-04"), "fund_name": name, "growth_of_1": 1.0}
            for name in [f"F{i}" for i in range(n_funds)]
        ]
        + [
            {"date": pd.Timestamp("2021-01-05"), "fund_name": name, "growth_of_1": 1.1}
            for name in [f"F{i}" for i in range(n_funds)]
        ]
    )
    allocations = {
        f"F{i}": share / 100.0 for i, share in enumerate(shares)
    }
    result = app.calculate_buy_and_hold_allocation(frame, allocations, 1000.0)
    assert result["ending_value"] == pytest.approx(1100.0)


def test_allocation_rejects_a_non_positive_initial_investment():
    with pytest.raises(ValueError, match="greater than zero"):
        app.calculate_buy_and_hold_allocation(
            _two_fund_fixture(), {"A": 0.5, "B": 0.5}, 0.0
        )


def test_default_three_fund_allocation_handles_the_real_mixed_calendar():
    """Regression test for the real default: crypto starts three days earlier."""
    returns = app.load_fund_returns()
    available = set(returns["fund_name"].unique())
    assert set(DEFAULT_ALLOCATION_FUNDS) <= available

    result = app.calculate_buy_and_hold_allocation(
        returns, dict.fromkeys(DEFAULT_ALLOCATION_FUNDS, 1 / 3), 10_000.0
    )
    timeline = result["timeline"]

    # Crypto has 2021-01-01..03 but the equity-calendar funds do not, so the
    # common start is the equity start and nothing is back-filled into January 1-3.
    assert result["common_start"] == pd.Timestamp("2021-01-04")
    assert timeline["date"].min() == pd.Timestamp("2021-01-04")
    assert not ((timeline["date"] >= "2021-01-01") & (timeline["date"] <= "2021-01-03")).any()

    # The union calendar runs on to the crypto fund's final day.
    assert result["end_date"] == pd.Timestamp("2023-12-31")
    assert timeline["portfolio_nav"].notna().all()
    assert timeline["portfolio_value"].notna().all()
    assert np.isfinite(result["ending_value"])
    assert np.isfinite(result["cumulative_return"])
    assert np.isfinite(result["max_drawdown"])


def test_default_allocation_holds_equity_navs_flat_on_the_final_crypto_only_days():
    returns = app.load_fund_returns()
    equity_like = ["Equity-Only Minimum Variance", "Combined Maximum Sharpe"]
    for fund in equity_like:
        series = returns.loc[returns["fund_name"] == fund].set_index("date")["growth_of_1"]
        assert series.index.max() == pd.Timestamp("2023-12-29")

    result = app.calculate_buy_and_hold_allocation(
        returns, dict.fromkeys(DEFAULT_ALLOCATION_FUNDS, 1 / 3), 10_000.0
    )
    timeline = result["timeline"].set_index("date")
    # 2023-12-30 and 12-31 exist only because the crypto fund still trades;
    # the portfolio must still be defined on them.
    for tail_date in ["2023-12-30", "2023-12-31"]:
        assert pd.Timestamp(tail_date) in timeline.index
        assert np.isfinite(timeline.loc[pd.Timestamp(tail_date), "portfolio_nav"])
    # Crypto keeps moving, so the portfolio NAV is not frozen over those days.
    assert timeline["portfolio_nav"].iloc[-1] != timeline["portfolio_nav"].iloc[-3]


# ============================================================================
# Allocation weight state when the fund selection changes
# ============================================================================


def test_shrinking_the_selection_drops_stale_weights_and_re_equalises():
    drop, weights = app.plan_allocation_weight_state(
        ["A", "B"],
        ["A", "B", "C"],
        ["allocation_A", "allocation_B", "allocation_C"],
    )
    # Without this, two stale 33.33% entries would remain and total 66.66%.
    assert drop == ["allocation_C"]
    assert weights == {"allocation_A": 50.0, "allocation_B": 50.0}
    assert sum(weights.values()) == pytest.approx(100.0)


def test_growing_the_selection_re_equalises_to_one_hundred():
    _drop, weights = app.plan_allocation_weight_state(
        ["A", "B", "C", "D"],
        ["A", "B", "C"],
        ["allocation_A", "allocation_B", "allocation_C"],
    )
    # Without this, three 33.33% entries plus a new default would total 125%.
    assert len(weights) == 4
    assert sum(weights.values()) == pytest.approx(100.0)


def test_unchanged_selection_leaves_manual_weights_alone():
    drop, weights = app.plan_allocation_weight_state(
        ["A", "B", "C"],
        ["A", "B", "C"],
        ["allocation_A", "allocation_B", "allocation_C"],
    )
    assert drop == []
    assert weights == {}


def test_force_reset_re_equalises_even_without_a_selection_change():
    _drop, weights = app.plan_allocation_weight_state(
        ["A", "B", "C"], ["A", "B", "C"], [], force_reset=True
    )
    assert sum(weights.values()) == pytest.approx(100.0)
    assert set(weights) == {"allocation_A", "allocation_B", "allocation_C"}


def test_removing_then_re_adding_a_fund_does_not_restore_its_old_weight():
    # Remove C: its stored weight must be dropped, not parked for later.
    drop, _weights = app.plan_allocation_weight_state(
        ["A", "B"], ["A", "B", "C"], ["allocation_A", "allocation_B", "allocation_C"]
    )
    assert "allocation_C" in drop
    # Re-add C: every fund is re-equalised, so C comes back on an equal share
    # (33.33 or the 33.34 rounding remainder) rather than a stale earlier value.
    _drop, weights = app.plan_allocation_weight_state(
        ["A", "B", "C"], ["A", "B"], ["allocation_A", "allocation_B"]
    )
    assert weights["allocation_C"] in {33.33, 33.34}
    assert sum(weights.values()) == pytest.approx(100.0)


def _allocation_page() -> AppTest:
    at = run_app()
    at.sidebar.radio[0].set_value("Allocation Lab").run()
    return at


def _allocation_weights(at: AppTest) -> list[float]:
    return [
        element.value
        for element in at.number_input
        if element.label != "Initial Investment ($)"
    ]


def test_app_default_three_fund_allocation_totals_one_hundred():
    at = _allocation_page()
    assert not at.exception
    weights = _allocation_weights(at)
    assert len(weights) == 3
    assert sum(weights) == pytest.approx(100.0)
    assert not [e.value for e in at.error if "add up to" in e.value]


def test_app_switching_three_funds_to_two_re_equalises_without_an_error():
    at = _allocation_page()
    selected = at.multiselect[0].value
    at.multiselect[0].set_value(selected[:2]).run()
    assert not at.exception

    weights = _allocation_weights(at)
    assert len(weights) == 2
    assert sum(weights) == pytest.approx(100.0)
    assert not [e.value for e in at.error if "add up to" in e.value]


def test_app_switching_three_funds_to_four_re_equalises_without_an_error():
    at = _allocation_page()
    available = at.multiselect[0].options
    selected = list(at.multiselect[0].value)
    extra = next(name for name in available if name not in selected)
    at.multiselect[0].set_value([*selected, extra]).run()
    assert not at.exception

    weights = _allocation_weights(at)
    assert len(weights) == 4
    assert sum(weights) == pytest.approx(100.0)
    assert not [e.value for e in at.error if "add up to" in e.value]


def test_app_manual_weights_survive_a_rerun_when_the_selection_is_unchanged():
    at = _allocation_page()

    # A deliberate non-equal split that still totals 100.
    for index, value in enumerate([50.0, 30.0, 20.0]):
        inputs = [e for e in at.number_input if e.label != "Initial Investment ($)"]
        inputs[index].set_value(value).run()
    assert _allocation_weights(at) == pytest.approx([50.0, 30.0, 20.0])

    # Rerun through an unrelated widget: the selection has not changed, so the
    # typed weights must not be snapped back to equal shares.
    investment = next(e for e in at.number_input if e.label == "Initial Investment ($)")
    investment.set_value(12_000.0).run()
    assert not at.exception

    assert _allocation_weights(at) == pytest.approx([50.0, 30.0, 20.0])
    assert not [e.value for e in at.error if "add up to" in e.value]


def test_app_reset_button_restores_equal_weights():
    at = _allocation_page()
    inputs = [e for e in at.number_input if e.label != "Initial Investment ($)"]
    inputs[0].set_value(80.0).run()
    assert _allocation_weights(at)[0] == pytest.approx(80.0)

    reset = next(b for b in at.button if "Reset" in b.label)
    reset.click().run()
    assert not at.exception
    weights = _allocation_weights(at)
    assert sum(weights) == pytest.approx(100.0)
    assert weights[0] != pytest.approx(80.0)


# ============================================================================
# Financial tables keep numeric dtypes so header sorting stays numeric
# ============================================================================


def test_fund_comparison_table_keeps_numeric_columns_matching_the_csv():
    metrics = app.load_performance_metrics()
    table = app.build_fund_comparison_table(metrics)
    for column in ["Annual Return", "Annual Volatility", "Sharpe Ratio", "Maximum Drawdown"]:
        assert pd.api.types.is_numeric_dtype(table[column]), column

    row = table.loc[table["Fund"] == "Combined Maximum Sharpe"].iloc[0]
    source = metrics.loc[metrics["fund_name"] == "Combined Maximum Sharpe"].iloc[0]
    assert row["Annual Return"] == pytest.approx(source["annual_return"] * 100)
    assert row["Annual Volatility"] == pytest.approx(source["annual_volatility"] * 100)
    assert row["Sharpe Ratio"] == pytest.approx(source["sharpe_ratio"])
    assert row["Maximum Drawdown"] == pytest.approx(source["max_drawdown"] * 100)


def test_holdings_table_keeps_a_numeric_target_weight_matching_the_csv():
    holdings = app.load_current_holdings()
    fund = holdings.loc[holdings["fund_name"] == "Combined Maximum Sharpe"]
    table = app.build_holdings_table(fund)
    assert pd.api.types.is_numeric_dtype(table["Target Weight"])
    assert len(table) <= 10
    # Descending, and matching the source weights.
    assert table["Target Weight"].is_monotonic_decreasing
    assert table["Target Weight"].max() == pytest.approx(fund["target_weight"].max() * 100)


def test_allocation_table_keeps_a_numeric_initial_allocation():
    table = app.build_allocation_table({"Fund A": 33.33, "Fund B": 66.67})
    assert pd.api.types.is_numeric_dtype(table["Initial Allocation"])
    assert table["Initial Allocation"].sum() == pytest.approx(100.0)


def test_fusion_comparison_table_keeps_numeric_columns_matching_the_csv():
    summary = app.load_fusion_summary()
    table = app.build_fusion_comparison_table(summary)
    turnover_column = "Average Rebalance Turnover (Excluding Initial)"
    for column in [
        "Annual Return", "Annual Volatility", "Sharpe Ratio", "Maximum Drawdown",
        turnover_column,
    ]:
        assert pd.api.types.is_numeric_dtype(table[column]), column

    base = summary.loc[summary["variant"] == "Equity-Only Minimum Variance"].iloc[0]
    row = table.loc[table["Variant"] == "Equity-Only Minimum Variance"].iloc[0]
    assert row["Annual Return"] == pytest.approx(base["annual_return"] * 100)
    assert row["Sharpe Ratio"] == pytest.approx(base["sharpe_ratio"])
    assert row[turnover_column] == pytest.approx(
        base["avg_rebalance_turnover_ex_initial"] * 100
    )


def test_turnover_column_label_states_that_the_initial_build_is_excluded():
    table = app.build_fusion_comparison_table(app.load_fusion_summary())
    assert "Average Rebalance Turnover (Excluding Initial)" in table.columns
    assert "Average Rebalance Turnover" not in table.columns


def test_no_financial_table_column_is_pre_formatted_text():
    """Percent/ratio columns must not arrive as strings like '5.20%'."""
    builders = [
        app.build_fund_comparison_table(app.load_performance_metrics()),
        app.build_fusion_comparison_table(app.load_fusion_summary()),
        app.build_holdings_table(app.load_current_holdings()),
        app.build_allocation_table({"A": 50.0, "B": 50.0}),
    ]
    for table in builders:
        for column in table.columns:
            if table[column].dtype == object:
                assert not table[column].astype(str).str.contains("%").any(), column


# ============================================================================
# Growth of $1 uses a linear axis
# ============================================================================


def test_growth_of_one_chart_uses_a_linear_axis_and_the_selected_fund_data():
    returns = app.load_fund_returns()
    fund = "Combined Maximum Sharpe"
    series = returns.loc[returns["fund_name"] == fund].sort_values("date")

    figure = app.make_fund_time_series_chart(series, fund, "#007C89", "Growth of $1")
    yaxis = figure.layout.yaxis
    # An unlabelled log axis would misrepresent the shape of the curve.
    assert yaxis.type != "log"
    assert yaxis.title.text == "Growth of $1"
    np.testing.assert_allclose(
        figure.data[0].y, series["growth_of_1"].to_numpy()
    )


def test_drawdown_chart_still_renders_percentages_on_a_linear_axis():
    returns = app.load_fund_returns()
    fund = "Combined Maximum Sharpe"
    series = returns.loc[returns["fund_name"] == fund].sort_values("date")
    figure = app.make_fund_time_series_chart(series, fund, "#007C89", "Drawdown")
    assert figure.layout.yaxis.type != "log"
    assert figure.layout.yaxis.title.text == "Drawdown (%)"
    np.testing.assert_allclose(
        figure.data[0].y, series["drawdown"].to_numpy() * 100
    )


def test_equal_weight_description_avoids_claiming_no_estimation_error():
    description = app._method_description("equal_weight")
    assert "no estimation error" not in description
    assert "parameter-estimation risk" in description


# ============================================================================
# 16.3 Sentiment display helpers
# ============================================================================


def test_rolling_mean_advances_by_valid_observations_and_keeps_gaps_missing():
    index = pd.date_range("2021-01-04", periods=30, freq="B")
    values = pd.Series(np.arange(30, dtype=float), index=index)
    values.iloc[10] = np.nan

    rolled = app.rolling_mean_of_valid_observations(values, window=5)

    # The missing day stays missing - the smoother never invents a reading.
    assert pd.isna(rolled.iloc[10])
    # No carry-forward: forward filling WOULD have produced a value there.
    assert pd.notna(rolled.ffill().iloc[10])
    # The first four valid observations cannot fill a 5-observation window.
    assert rolled.iloc[:4].isna().all()
    # Window 5 over the first five valid values (0..4) -> mean 2.0.
    assert rolled.iloc[4] == pytest.approx(2.0)
    assert rolled.index.equals(values.index)


def test_sector_sentiment_distinguishes_latest_trading_day_from_latest_reading():
    sentiment = app.load_sector_sentiment()
    sector = sentiment.loc[sentiment["sector"] == "Tech"].sort_values("date")
    latest_trading_date = sector["date"].max()
    observed = sector.dropna(subset=["extended_z_expanding"])
    latest_available = observed["date"].max()
    # The two concepts must be tracked separately, and a reading can never
    # post-date the last trading day.
    assert latest_available <= latest_trading_date


def test_app_uses_the_extended_signal_and_offers_no_baseline_selector():
    source = APP_PATH.read_text()
    assert "extended_z_expanding" in source
    assert "extended_band" in source
    assert "baseline_z_expanding" not in source
    assert "Baseline" not in source


# ============================================================================
# 16.4 Dynamic copy
# ============================================================================


def test_fund_universe_summary_follows_the_dataframe():
    metrics = app.load_performance_metrics()
    real = app.summarise_fund_universe(metrics)
    assert real["n_funds"] == metrics["fund_name"].nunique()
    assert real["n_families"] == metrics["fund_family"].nunique()
    assert real["n_methods"] == metrics["method"].nunique()

    trimmed = metrics.head(2)
    changed = app.summarise_fund_universe(trimmed)
    assert changed["n_funds"] == 2
    assert changed["n_funds"] != real["n_funds"]


def test_coverage_summary_follows_the_dataframe():
    coverage = app.load_fusion_coverage()
    real = app.summarise_fusion_coverage(coverage)
    assert real["n_formations"] == len(coverage)
    assert real["mean_coverage"] == pytest.approx(coverage["coverage_ratio"].mean())
    assert real["inactive_count"] == int((~coverage["tilt_active"].astype(bool)).sum())

    synthetic = pd.DataFrame(
        {
            "formation_date": pd.to_datetime(["2021-01-29", "2021-02-26"]),
            "effective_date": pd.to_datetime(["2021-02-01", "2021-03-01"]),
            "coverage_ratio": [0.25, 0.75],
            "tilt_active": [False, True],
        }
    )
    changed = app.summarise_fusion_coverage(synthetic)
    assert changed["n_formations"] == 2
    assert changed["mean_coverage"] == pytest.approx(0.5)
    assert changed["inactive_count"] == 1


def test_fusion_commentary_direction_follows_the_data():
    base = "Equity-Only Minimum Variance"
    better = pd.DataFrame(
        [
            {
                "variant": base, "annual_return": 0.05, "annual_volatility": 0.12,
                "sharpe_ratio": 0.40, "max_drawdown": -0.15,
                "avg_rebalance_turnover_ex_initial": 0.15,
            },
            {
                "variant": "Extended Sentiment Momentum", "annual_return": 0.09,
                "annual_volatility": 0.10, "sharpe_ratio": 0.90, "max_drawdown": -0.10,
                "avg_rebalance_turnover_ex_initial": 0.30,
            },
        ]
    )
    text = app.describe_fusion_variant(better, "Extended Sentiment Momentum")
    assert "a higher annualised return" in text
    assert "a higher Sharpe" in text
    assert "less volatile" in text
    assert "a shallower maximum" in text
    assert "2.0x" in text

    worse = better.copy()
    worse.loc[1, ["annual_return", "sharpe_ratio", "annual_volatility", "max_drawdown"]] = [
        0.01, 0.10, 0.20, -0.30
    ]
    flipped = app.describe_fusion_variant(worse, "Extended Sentiment Momentum")
    assert "a lower annualised return" in flipped
    assert "a lower Sharpe" in flipped
    assert "more volatile" in flipped
    assert "a deeper maximum" in flipped
    # Neutral language only - the app never crowns a winner.
    for verdict in ("winner", "recommend", "best investment", "guaranteed"):
        assert verdict not in flipped.lower()


def test_no_hardcoded_result_numbers_in_the_app_source():
    source = APP_PATH.read_text()
    for literal in ["66.3", "12 funds", "3 asset universes", "4 methods", "2.2x", "2021–2023"]:
        assert literal not in source, f"hard-coded result value found: {literal}"


# ============================================================================
# 16.5 AppTest: pages and interactions
# ============================================================================


def test_starter_placeholders_are_gone():
    source = APP_PATH.read_text()
    assert "TODO" not in source
    assert "starter" not in source.lower()


def test_default_page_is_fund_explorer_and_runs_without_exception():
    at = run_app()
    assert not at.exception
    titles = [element.value for element in at.title]
    assert "RiskBridge Funds" in titles
    # Other pages must not be rendered at the same time.
    assert "Allocation Lab" not in titles
    assert "Sentiment & Fusion" not in titles


def test_sidebar_navigation_switches_to_allocation_lab_only():
    at = run_app()
    at.sidebar.radio[0].set_value("Allocation Lab").run()
    assert not at.exception
    titles = [element.value for element in at.title]
    assert "Allocation Lab" in titles
    assert "Sentiment & Fusion" not in titles
    # Fund Explorer's fact-sheet control is not present on this page.
    labels = [element.label for element in at.selectbox]
    assert "Select a fund" not in labels


def test_sidebar_navigation_switches_to_sentiment_and_fusion_only():
    at = run_app()
    at.sidebar.radio[0].set_value("Sentiment & Fusion").run()
    assert not at.exception
    titles = [element.value for element in at.title]
    assert "Sentiment & Fusion" in titles
    assert "Allocation Lab" not in titles


def test_fund_explorer_filters_and_fact_sheet_controls_work():
    at = run_app()
    assert not at.exception
    labels = [element.label for element in at.selectbox]
    assert "Fund Family" in labels
    assert "Portfolio Method" in labels
    assert "Select a fund" in labels

    family = next(e for e in at.selectbox if e.label == "Fund Family")
    family.set_value("Equity-Only").run()
    assert not at.exception

    fund_selector = next(e for e in at.selectbox if e.label == "Select a fund")
    fund_selector.set_value("Equity-Only HRP").run()
    assert not at.exception


def test_fact_sheet_growth_and_drawdown_radio_switches():
    at = run_app()
    chart_radio = next(r for r in at.radio if r.label == "Chart")
    assert chart_radio.value == "Growth of $1"
    chart_radio.set_value("Drawdown").run()
    assert not at.exception
    assert next(r for r in at.radio if r.label == "Chart").value == "Drawdown"


def test_allocation_lab_default_three_funds_render_a_valid_result():
    at = run_app()
    at.sidebar.radio[0].set_value("Allocation Lab").run()
    assert not at.exception

    selected = at.multiselect[0].value
    assert 2 <= len(selected) <= 4
    assert set(selected) == set(DEFAULT_ALLOCATION_FUNDS)

    # Equal thirds must be accepted as a valid 100% configuration.
    assert not at.error, [element.value for element in at.error]
    metric_labels = [element.label for element in at.metric]
    for label in ["Initial Investment", "Ending Value", "Cumulative Return", "Maximum Drawdown"]:
        assert label in metric_labels

    caption_text = " ".join(element.value for element in at.caption)
    assert "04 Jan 2021" in caption_text
    for value in [element.value for element in at.metric]:
        assert "nan" not in str(value).lower()


def test_allocation_lab_rejects_an_allocation_that_does_not_total_one_hundred():
    at = run_app()
    at.sidebar.radio[0].set_value("Allocation Lab").run()
    at.number_input[0].set_value(10.0).run()
    assert not at.exception
    assert any("add up to" in element.value for element in at.error)


def test_sentiment_subviews_switch_and_both_render():
    at = run_app()
    at.sidebar.radio[0].set_value("Sentiment & Fusion").run()
    assert not at.exception

    view = next(r for r in at.radio if r.label == "View")
    assert view.value == "Sector Sentiment"
    assert any(e.label == "Select an equity sector" for e in at.selectbox)

    view.set_value("Sentiment Tilt").run()
    assert not at.exception
    assert next(r for r in at.radio if r.label == "View").value == "Sentiment Tilt"
    # The sector selector belongs to the other sub-view and must be gone.
    assert not any(e.label == "Select an equity sector" for e in at.selectbox)


def test_sector_selector_can_be_changed():
    at = run_app()
    at.sidebar.radio[0].set_value("Sentiment & Fusion").run()
    sector = next(e for e in at.selectbox if e.label == "Select an equity sector")
    other = next(option for option in sector.options if option != sector.value)
    sector.set_value(other).run()
    assert not at.exception


def test_sentiment_tilt_shows_three_variants_and_no_baseline_fusion():
    at = run_app()
    at.sidebar.radio[0].set_value("Sentiment & Fusion").run()
    next(r for r in at.radio if r.label == "View").set_value("Sentiment Tilt").run()
    assert not at.exception

    rendered = " ".join(element.value for element in at.markdown)
    rendered += " ".join(element.value for element in at.caption)
    for variant in [
        "Equity-Only Minimum Variance",
        "Extended Sentiment Momentum",
        "Extended Sentiment Contrarian",
    ]:
        assert variant in rendered
    assert "Baseline Sentiment" not in rendered
    assert "Baseline Momentum" not in rendered
    assert "Baseline Contrarian" not in rendered


# ============================================================================
# 16.6 Forbidden dependencies
# ============================================================================


@pytest.mark.parametrize(
    "forbidden",
    [
        "data_access", "load_equity_prices", "load_news_headlines", "nltk",
        "finvader", "run_part_b", "parquet", ".png", "st.tabs",
        "run_walk_forward_backtest", "build_fund_universe",
    ],
)
def test_app_source_avoids_forbidden_dependencies(forbidden):
    assert forbidden not in APP_PATH.read_text()


def test_app_source_has_no_absolute_paths():
    source = APP_PATH.read_text()
    assert "/Users/" not in source
    assert "C:\\" not in source


def test_app_reads_only_the_seven_permitted_csvs():
    source = APP_PATH.read_text()
    permitted = {pathlib.Path(name).name for name in REQUIRED_CSVS}
    referenced = {
        token.strip('"').strip("'")
        for token in source.replace("(", " ").replace(")", " ").split()
        if token.strip('"').strip("'").endswith(".csv")
    }
    assert referenced <= permitted, f"unexpected CSV reference(s): {referenced - permitted}"
    # Explicitly excluded detail files.
    for excluded in ["ticker_sentiment_z.csv", "fusion_weights.csv", "fund_weights.csv"]:
        assert excluded not in source
