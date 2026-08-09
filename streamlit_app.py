"""RiskBridge Funds - investor-facing Streamlit app (Station 4).

Reads only the precomputed CSVs under results/. It never loads raw data,
never scores sentiment, and never re-runs an optimisation or a backtest, so a
cold start on a basic machine stays fast.

Three pages, dispatched from a sidebar radio so only the selected page's data
and charts are ever built:

    Fund Explorer      - compare funds, then open one fund's fact sheet
    Allocation Lab     - illustrative buy-and-hold allocation across 2-4 funds
    Sentiment & Fusion - sector news sentiment, and the sentiment tilt result

Every descriptive number shown to the user is computed from the loaded data,
so the copy stays correct if the underlying results are rebuilt.
"""
from __future__ import annotations

import pathlib

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ============================================================================
# Paths
# ============================================================================

ROOT = pathlib.Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DATA = RESULTS / "data"
TABLES = RESULTS / "tables"

FUND_RETURNS_PATH = DATA / "fund_returns.csv"
FUSION_RETURNS_PATH = DATA / "fusion_returns.csv"
SECTOR_SENTIMENT_PATH = DATA / "sector_sentiment_index.csv"
PERFORMANCE_METRICS_PATH = TABLES / "performance_metrics.csv"
CURRENT_HOLDINGS_PATH = TABLES / "current_holdings.csv"
FUSION_SUMMARY_PATH = TABLES / "fusion_before_vs_after.csv"
FUSION_COVERAGE_PATH = TABLES / "ticker_sentiment_coverage_by_formation_date.csv"


# ============================================================================
# Labels, exact colours and display configuration
#
# The hex values match the figures in the written report exactly, so a colour
# means the same thing in the app and on the page.
# ============================================================================

APP_TITLE = "RiskBridge Funds"
APP_TAGLINE = (
    "Systematic funds informed by market data and financial-news sentiment."
)
APP_DISCLAIMER = (
    "Out-of-sample evidence · Illustrative only · Not financial advice"
)

FAMILY_COLORS = {
    "Equity-Only": "#1F3A5F",
    "Crypto-Only": "#C99700",
    "Combined": "#007C89",
}

METHOD_SYMBOLS = {
    "equal_weight": "circle",
    "minimum_variance": "square",
    "maximum_sharpe": "triangle-up",
    "hrp": "diamond",
}

METHOD_LABELS = {
    "equal_weight": "Equal Weight",
    "minimum_variance": "Minimum Variance",
    "maximum_sharpe": "Maximum Sharpe",
    "hrp": "HRP",
}

FUSION_STYLES = {
    "Equity-Only Minimum Variance": {"color": "#4A5568", "dash": "solid", "width": 2.6},
    "Extended Sentiment Momentum": {"color": "#B23A48", "dash": "solid", "width": 1.8},
    "Extended Sentiment Contrarian": {"color": "#2E86C1", "dash": "dash", "width": 1.8},
}
FUSION_BASE_VARIANT = "Equity-Only Minimum Variance"

SENTIMENT_DAILY_COLOR = "#B8AEA7"
SENTIMENT_ROLLING_COLOR = "#007C89"
SENTIMENT_ZERO_COLOR = "#4B5563"
SENTIMENT_THRESHOLD_COLOR = "#D8DDE6"

ALLOCATION_COLOR = "#007C89"

DEFAULT_FACT_SHEET_FUND = "Combined Maximum Sharpe"
DEFAULT_ALLOCATION_FUNDS = [
    "Equity-Only Minimum Variance",
    "Crypto-Only HRP",
    "Combined Maximum Sharpe",
]
DEFAULT_SECTOR = "Tech"
ALLOCATION_TOLERANCE = 1e-6
ROLLING_WINDOW = 21
EXPANDING_MIN_PERIODS = 60

PLOT_LAYOUT = {
    "template": "plotly_white",
    "hovermode": "x unified",
    "margin": {"l": 60, "r": 30, "t": 60, "b": 50},
    "height": 430,
    "legend": {
        "orientation": "h",
        "yanchor": "bottom",
        "y": 1.02,
        "xanchor": "left",
        "x": 0,
    },
}


class AppDataError(RuntimeError):
    """Raised when a required results file is missing or has the wrong schema."""


# ============================================================================
# Cached CSV loaders and schema validation
# ============================================================================


def _display_path(path: pathlib.Path) -> str:
    """Project-relative path for messages, falling back to the full path.

    An error message must never fail to render because the path happens to sit
    outside the project directory.
    """
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _read_results_csv(
    path: pathlib.Path,
    required_columns: list[str],
    date_columns: list[str] | None = None,
    sort_by: list[str] | None = None,
) -> pd.DataFrame:
    """Read one results CSV, validating that it is present and well-formed."""
    shown = _display_path(path)
    if not path.exists():
        raise AppDataError(
            f"Required results file is missing: {shown}. "
            "Rebuild the project results before starting the app."
        )
    frame = pd.read_csv(path, parse_dates=date_columns or [])
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise AppDataError(
            f"{shown} is missing expected column(s): {', '.join(missing)}."
        )
    if frame.empty:
        raise AppDataError(f"{shown} contains no rows.")
    for column in date_columns or []:
        if frame[column].isna().any():
            raise AppDataError(f"{shown} has unparseable values in '{column}'.")
    if sort_by:
        frame = frame.sort_values(sort_by).reset_index(drop=True)
    return frame


@st.cache_data(show_spinner=False)
def load_fund_returns() -> pd.DataFrame:
    return _read_results_csv(
        FUND_RETURNS_PATH,
        ["date", "fund_family", "method", "fund_name", "growth_of_1", "drawdown"],
        date_columns=["date"],
        sort_by=["fund_name", "date"],
    )


@st.cache_data(show_spinner=False)
def load_performance_metrics() -> pd.DataFrame:
    return _read_results_csv(
        PERFORMANCE_METRICS_PATH,
        [
            "fund_name", "fund_family", "method", "oos_start", "oos_end",
            "annual_return", "annual_volatility", "sharpe_ratio", "max_drawdown",
        ],
        date_columns=["oos_start", "oos_end"],
        sort_by=["fund_name"],
    )


@st.cache_data(show_spinner=False)
def load_current_holdings() -> pd.DataFrame:
    return _read_results_csv(
        CURRENT_HOLDINGS_PATH,
        ["formation_date", "effective_date", "fund_name", "ticker", "target_weight"],
        date_columns=["formation_date", "effective_date"],
        sort_by=["fund_name", "target_weight"],
    )


@st.cache_data(show_spinner=False)
def load_sector_sentiment() -> pd.DataFrame:
    return _read_results_csv(
        SECTOR_SENTIMENT_PATH,
        [
            "date", "sector", "extended_z_expanding", "extended_band",
            "headline_count", "active_ticker_count", "coverage_ratio",
        ],
        date_columns=["date"],
        sort_by=["sector", "date"],
    )


@st.cache_data(show_spinner=False)
def load_fusion_returns() -> pd.DataFrame:
    return _read_results_csv(
        FUSION_RETURNS_PATH,
        ["date", "variant", "growth_of_1"],
        date_columns=["date"],
        sort_by=["variant", "date"],
    )


@st.cache_data(show_spinner=False)
def load_fusion_summary() -> pd.DataFrame:
    return _read_results_csv(
        FUSION_SUMMARY_PATH,
        [
            "variant", "annual_return", "annual_volatility", "sharpe_ratio",
            "max_drawdown", "avg_rebalance_turnover_ex_initial",
        ],
    )


@st.cache_data(show_spinner=False)
def load_fusion_coverage() -> pd.DataFrame:
    return _read_results_csv(
        FUSION_COVERAGE_PATH,
        ["formation_date", "effective_date", "coverage_ratio", "tilt_active"],
        date_columns=["formation_date", "effective_date"],
        sort_by=["formation_date"],
    )


def load_or_stop(loader):
    """Call a loader, turning a data problem into a readable message.

    Keeps a missing or malformed results file from surfacing as a Python
    traceback in the browser.
    """
    try:
        return loader()
    except AppDataError as exc:
        st.error(str(exc))
        st.stop()


# ============================================================================
# Formatting helpers
# ============================================================================


def format_percent(value: float, decimals: int = 2) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value * 100:.{decimals}f}%"


def format_ratio(value: float, decimals: int = 2) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.{decimals}f}"


def format_dollars(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"${value:,.2f}"


def format_date(value) -> str:
    if pd.isna(value):
        return "n/a"
    return pd.Timestamp(value).strftime("%d %b %Y")


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, str(method).replace("_", " ").title())


# ============================================================================
# Pure calculations (no Streamlit state, so they are directly unit-testable)
# ============================================================================


def calculate_buy_and_hold_allocation(
    fund_returns: pd.DataFrame,
    allocations: dict[str, float],
    initial_investment: float,
) -> dict[str, object]:
    """Combine fund NAVs into one buy-and-hold portfolio.

    Weights apply once, at the common start date; the portfolio is then left
    alone. Compounding a fixed-weight average of daily returns instead would
    silently assume the investor rebalances across funds every single day,
    which is a different (and much more actively traded) product.

    Funds run on different calendars: crypto trades seven days a week while
    equity funds do not. The portfolio is evaluated on the union of the
    selected funds' dates, and a fund's NAV is carried forward across a day it
    does not trade - that is the last observable valuation of a holding the
    investor still owns, not a stale signal being reused. Nothing is filled
    backwards, so no fund contributes before it actually starts.
    """
    if not 2 <= len(allocations) <= 4:
        raise ValueError("Select between 2 and 4 funds.")
    if any(weight < 0 for weight in allocations.values()):
        raise ValueError("Allocations cannot be negative.")
    total = sum(allocations.values())
    if abs(total - 1.0) > ALLOCATION_TOLERANCE:
        raise ValueError(
            f"Allocations must add up to 100%; they currently add up to {total * 100:.2f}%."
        )
    if not initial_investment > 0:
        raise ValueError("Initial investment must be greater than zero.")

    selected = list(allocations)
    missing = sorted(set(selected) - set(fund_returns["fund_name"].unique()))
    if missing:
        raise ValueError(f"Fund(s) not found in the results: {', '.join(missing)}.")

    subset = fund_returns.loc[
        fund_returns["fund_name"].isin(selected), ["date", "fund_name", "growth_of_1"]
    ]
    if subset.duplicated(["date", "fund_name"]).any():
        raise ValueError("Fund returns contain duplicate (date, fund) rows.")

    nav = subset.pivot(index="date", columns="fund_name", values="growth_of_1")
    nav = nav.sort_index()
    # Forward fill only. A fund's NAV before its own first observation stays
    # missing, so an earlier-starting fund cannot pull a later one backwards.
    nav = nav.ffill()

    complete = nav.dropna(how="any")
    if complete.empty:
        raise ValueError("The selected funds have no overlapping history.")
    common_start = complete.index.min()

    trimmed = nav.loc[common_start:]
    normalised = trimmed / trimmed.iloc[0]

    portfolio_nav = sum(
        allocations[fund] * normalised[fund] for fund in selected
    )
    portfolio_nav.name = "portfolio_nav"
    dollar_value = portfolio_nav * initial_investment
    drawdown = portfolio_nav / portfolio_nav.cummax() - 1.0

    timeline = pd.DataFrame(
        {
            "date": portfolio_nav.index,
            "portfolio_nav": portfolio_nav.to_numpy(),
            "portfolio_value": dollar_value.to_numpy(),
            "drawdown": drawdown.to_numpy(),
        }
    )
    return {
        "timeline": timeline,
        "common_start": pd.Timestamp(common_start),
        "end_date": pd.Timestamp(portfolio_nav.index.max()),
        "initial_investment": float(initial_investment),
        "ending_value": float(dollar_value.iloc[-1]),
        "cumulative_return": float(portfolio_nav.iloc[-1] - 1.0),
        "max_drawdown": float(drawdown.min()),
    }


def equal_default_allocation_pct(n_funds: int) -> list[float]:
    """Near-equal default percentages that add up to exactly 100.

    Rounding 100/3 to two decimals three times gives 99.99, which the
    validator would (correctly) reject - so the page would open showing an
    error on its own default. The rounding remainder goes to the last fund so
    the default configuration is always immediately valid.
    """
    if n_funds < 1:
        raise ValueError("n_funds must be at least 1")
    shares = [round(100.0 / n_funds, 2)] * n_funds
    shares[-1] = round(100.0 - sum(shares[:-1]), 2)
    return shares


ALLOCATION_STATE_PREFIX = "allocation_"
ALLOCATION_SELECTION_STATE_KEY = "_allocation_selection"


def allocation_state_key(fund: str) -> str:
    return f"{ALLOCATION_STATE_PREFIX}{fund}"


def plan_allocation_weight_state(
    selected: list[str],
    previous_selection: list[str] | None,
    existing_keys: list[str],
    force_reset: bool = False,
) -> tuple[list[str], dict[str, float]]:
    """Decide which weight inputs to drop and which to reset to equal weights.

    Streamlit keeps a keyed widget's value in session state even after the
    widget stops being rendered. Without this, going from three funds to two
    would keep two stale 33.33% entries and total 66.66%, and going to four
    would total 125% - the page would open on an error the user did not cause.

    Returns ``(keys_to_drop, weights_to_set)``. Kept as a pure function so the
    decision can be tested without a Streamlit session.
    """
    selection_changed = previous_selection != selected

    keys_to_drop: list[str] = []
    if selection_changed:
        # Forget funds that are no longer selected, so re-adding one later
        # starts from a fresh equal weight instead of a stale earlier value.
        selected_keys = {allocation_state_key(fund) for fund in selected}
        keys_to_drop = [
            key
            for key in existing_keys
            if key.startswith(ALLOCATION_STATE_PREFIX) and key not in selected_keys
        ]

    weights_to_set: dict[str, float] = {}
    if selection_changed or force_reset:
        shares = equal_default_allocation_pct(len(selected))
        weights_to_set = {
            allocation_state_key(fund): share for fund, share in zip(selected, shares)
        }
    return keys_to_drop, weights_to_set


def sync_allocation_weight_state(selected: list[str], *, force_reset: bool = False) -> None:
    """Apply ``plan_allocation_weight_state`` to the live session state.

    A selection that has not changed is left alone, so a weight the user typed
    survives an ordinary rerun.
    """
    keys_to_drop, weights_to_set = plan_allocation_weight_state(
        selected,
        st.session_state.get(ALLOCATION_SELECTION_STATE_KEY),
        list(st.session_state.keys()),
        force_reset=force_reset,
    )
    for key in keys_to_drop:
        del st.session_state[key]
    for key, share in weights_to_set.items():
        st.session_state[key] = share
    # Any fund without state yet (for example after a manual state edit) gets
    # a sensible starting value rather than crashing the widget.
    for fund, share in zip(selected, equal_default_allocation_pct(len(selected))):
        st.session_state.setdefault(allocation_state_key(fund), share)
    st.session_state[ALLOCATION_SELECTION_STATE_KEY] = list(selected)


def rolling_mean_of_valid_observations(
    series: pd.Series, window: int = ROLLING_WINDOW
) -> pd.Series:
    """Rolling mean over the latest ``window`` VALID observations.

    Dropping missing values first means the window advances by observation
    count rather than by calendar position, so it never bridges a no-news gap
    with stale history. Reindexing afterwards restores NaN on exactly the
    dates the daily series is missing: the smoother is for display only and
    must not invent a reading where there was none.
    """
    valid = series.dropna()
    rolled = valid.rolling(window=window, min_periods=window).mean()
    return rolled.reindex(series.index)


def summarise_fund_universe(performance_metrics: pd.DataFrame) -> dict[str, object]:
    """Headline counts for the Fund Explorer caption, straight from the data."""
    return {
        "n_funds": int(performance_metrics["fund_name"].nunique()),
        "n_families": int(performance_metrics["fund_family"].nunique()),
        "n_methods": int(performance_metrics["method"].nunique()),
        "start": pd.Timestamp(performance_metrics["oos_start"].min()),
        "end": pd.Timestamp(performance_metrics["oos_end"].max()),
    }


def summarise_fusion_coverage(coverage: pd.DataFrame) -> dict[str, object]:
    """Coverage facts for the tilt commentary, computed from the coverage table."""
    return {
        "n_formations": int(len(coverage)),
        "mean_coverage": float(coverage["coverage_ratio"].mean()),
        "inactive_count": int((~coverage["tilt_active"].astype(bool)).sum()),
        "start": pd.Timestamp(coverage["formation_date"].min()),
        "end": pd.Timestamp(coverage["effective_date"].max()),
    }


def describe_fusion_variant(
    fusion_summary: pd.DataFrame, variant: str, base_variant: str = FUSION_BASE_VARIANT
) -> str:
    """Neutral sentence describing one tilt variant against the base fund.

    Every direction word is derived from the actual differences, so the text
    follows the data if the results are rebuilt. It deliberately does not name
    a winner or make a recommendation.
    """
    indexed = fusion_summary.set_index("variant")
    if variant not in indexed.index or base_variant not in indexed.index:
        return "Comparison unavailable for this variant."

    row, base = indexed.loc[variant], indexed.loc[base_variant]

    def direction(value: float, base_value: float, higher: str, lower: str) -> str:
        if pd.isna(value) or pd.isna(base_value):
            return "is not comparable"
        difference = value - base_value
        if abs(difference) < 1e-12:
            return "matches the base fund"
        return higher if difference > 0 else lower

    return_word = direction(
        row["annual_return"], base["annual_return"], "a higher", "a lower"
    )
    sharpe_word = direction(
        row["sharpe_ratio"], base["sharpe_ratio"], "a higher", "a lower"
    )
    vol_word = direction(
        row["annual_volatility"], base["annual_volatility"], "more", "less"
    )
    # A drawdown is negative, so a larger (less negative) value is the shallower one.
    drawdown_word = direction(
        row["max_drawdown"], base["max_drawdown"], "a shallower", "a deeper"
    )
    turnover_ratio = (
        row["avg_rebalance_turnover_ex_initial"]
        / base["avg_rebalance_turnover_ex_initial"]
        if base["avg_rebalance_turnover_ex_initial"]
        else float("nan")
    )
    turnover_text = (
        f"and traded {turnover_ratio:.1f}x the base fund's average rebalance turnover"
        if pd.notna(turnover_ratio)
        else "with turnover not comparable"
    )

    return (
        f"Over the evaluated out-of-sample window, **{variant}** delivered "
        f"{return_word} annualised return ({format_percent(row['annual_return'])} "
        f"vs {format_percent(base['annual_return'])}) and {sharpe_word} Sharpe ratio "
        f"({format_ratio(row['sharpe_ratio'])} vs {format_ratio(base['sharpe_ratio'])}) "
        f"than the untilted base fund, was {vol_word} volatile "
        f"({format_percent(row['annual_volatility'])} vs "
        f"{format_percent(base['annual_volatility'])}), had {drawdown_word} maximum "
        f"drawdown ({format_percent(row['max_drawdown'])} vs "
        f"{format_percent(base['max_drawdown'])}), {turnover_text}."
    )


# ============================================================================
# Table builders
#
# These return NUMERIC frames and a matching column configuration, rather than
# pre-formatted strings. Formatting a percentage into text would make the
# column sort alphabetically ("9.61%" above "23.50%") the moment a user clicks
# the header, so the number stays a number and only its display is formatted.
# Percentages are carried in percentage points so a printf format can render
# them directly while still sorting numerically.
# ============================================================================

PERCENT_FORMAT = "%.2f%%"
RATIO_FORMAT = "%.2f"
DATE_FORMAT = "DD MMM YYYY"


def build_fund_comparison_table(metrics: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Fund": metrics["fund_name"].to_numpy(),
            "Family": metrics["fund_family"].to_numpy(),
            "Method": metrics["method"].map(method_label).to_numpy(),
            "Annual Return": metrics["annual_return"].to_numpy() * 100.0,
            "Annual Volatility": metrics["annual_volatility"].to_numpy() * 100.0,
            "Sharpe Ratio": metrics["sharpe_ratio"].to_numpy(),
            "Maximum Drawdown": metrics["max_drawdown"].to_numpy() * 100.0,
        }
    )


def fund_comparison_column_config() -> dict[str, object]:
    return {
        "Annual Return": st.column_config.NumberColumn(
            "Annual Return", format=PERCENT_FORMAT
        ),
        "Annual Volatility": st.column_config.NumberColumn(
            "Annual Volatility", format=PERCENT_FORMAT
        ),
        "Sharpe Ratio": st.column_config.NumberColumn(
            "Sharpe Ratio", format=RATIO_FORMAT
        ),
        "Maximum Drawdown": st.column_config.NumberColumn(
            "Maximum Drawdown", format=PERCENT_FORMAT
        ),
    }


def build_holdings_table(holdings: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    ranked = holdings.sort_values("target_weight", ascending=False).head(top_n)
    return pd.DataFrame(
        {
            "Ticker": ranked["ticker"].to_numpy(),
            "Target Weight": ranked["target_weight"].to_numpy() * 100.0,
        }
    )


def holdings_column_config() -> dict[str, object]:
    return {
        "Target Weight": st.column_config.NumberColumn(
            "Target Weight", format=PERCENT_FORMAT
        )
    }


def build_allocation_table(allocations_pct: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Fund": list(allocations_pct),
            "Initial Allocation": [float(pct) for pct in allocations_pct.values()],
        }
    )


def allocation_column_config() -> dict[str, object]:
    return {
        "Initial Allocation": st.column_config.NumberColumn(
            "Initial Allocation", format=PERCENT_FORMAT
        )
    }


def build_fusion_comparison_table(summary: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Variant": summary["variant"].to_numpy(),
            "Annual Return": summary["annual_return"].to_numpy() * 100.0,
            "Annual Volatility": summary["annual_volatility"].to_numpy() * 100.0,
            "Sharpe Ratio": summary["sharpe_ratio"].to_numpy(),
            "Maximum Drawdown": summary["max_drawdown"].to_numpy() * 100.0,
            # The column shows avg_rebalance_turnover_ex_initial, so the label
            # says so rather than implying it includes the initial build.
            "Average Rebalance Turnover (Excluding Initial)": (
                summary["avg_rebalance_turnover_ex_initial"].to_numpy() * 100.0
            ),
        }
    )


def fusion_comparison_column_config() -> dict[str, object]:
    return {
        "Annual Return": st.column_config.NumberColumn(
            "Annual Return", format=PERCENT_FORMAT
        ),
        "Annual Volatility": st.column_config.NumberColumn(
            "Annual Volatility", format=PERCENT_FORMAT
        ),
        "Sharpe Ratio": st.column_config.NumberColumn(
            "Sharpe Ratio", format=RATIO_FORMAT
        ),
        "Maximum Drawdown": st.column_config.NumberColumn(
            "Maximum Drawdown", format=PERCENT_FORMAT
        ),
        "Average Rebalance Turnover (Excluding Initial)": st.column_config.NumberColumn(
            "Average Rebalance Turnover (Excluding Initial)", format=PERCENT_FORMAT
        ),
    }


def table_height(frame: pd.DataFrame, max_height: int = 460) -> int:
    """Compact height that grows with the row count up to a cap."""
    return min(38 * len(frame) + 38, max_height)


# ============================================================================
# Plotly chart builders
# ============================================================================


def _apply_layout(figure: go.Figure, title: str, **overrides) -> go.Figure:
    layout = dict(PLOT_LAYOUT)
    layout.update(overrides)
    figure.update_layout(title=title, **layout)
    return figure


def make_risk_return_chart(metrics: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    for family, family_rows in metrics.groupby("fund_family"):
        for method, rows in family_rows.groupby("method"):
            figure.add_trace(
                go.Scatter(
                    x=rows["annual_volatility"] * 100,
                    y=rows["annual_return"] * 100,
                    mode="markers",
                    name=f"{family} · {method_label(method)}",
                    marker={
                        "size": 13,
                        "color": FAMILY_COLORS.get(family, "#4A5568"),
                        "symbol": METHOD_SYMBOLS.get(method, "circle"),
                        "line": {"width": 1, "color": "#262A33"},
                    },
                    customdata=rows[
                        ["fund_name", "fund_family", "method", "sharpe_ratio", "max_drawdown"]
                    ].assign(method=rows["method"].map(method_label)).to_numpy(),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Family: %{customdata[1]}<br>"
                        "Method: %{customdata[2]}<br>"
                        "Annual Return: %{y:.2f}%<br>"
                        "Annual Volatility: %{x:.2f}%<br>"
                        "Sharpe Ratio: %{customdata[3]:.2f}<br>"
                        "Maximum Drawdown: %{customdata[4]:.2%}"
                        "<extra></extra>"
                    ),
                )
            )
    # Up to twelve family-method entries. A horizontal legend either wraps
    # onto the title or drifts across the low-return corner of the plot, so it
    # is parked in its own column to the right of the axes where it cannot
    # cover a data point.
    _apply_layout(
        figure,
        "Risk and return across available funds",
        hovermode="closest",
        height=520,
        margin={"l": 60, "r": 250, "t": 60, "b": 60},
        legend={
            "orientation": "v",
            "yanchor": "top",
            "y": 1.0,
            "xanchor": "left",
            "x": 1.02,
            "font": {"size": 11},
        },
    )
    figure.update_xaxes(title="Annual Volatility (%)")
    figure.update_yaxes(title="Annual Return (%)")
    return figure


def make_fund_time_series_chart(
    series: pd.DataFrame, fund_name: str, colour: str, mode: str
) -> go.Figure:
    figure = go.Figure()
    if mode == "Growth of $1":
        figure.add_trace(
            go.Scatter(
                x=series["date"],
                y=series["growth_of_1"],
                mode="lines",
                name=fund_name,
                line={"color": colour, "width": 2},
                hovertemplate="%{x|%d %b %Y}<br>Growth of $1: $%{y:.2f}<extra></extra>",
            )
        )
        _apply_layout(figure, f"Growth of $1 · {fund_name}")
        # Linear axis, matching the allocation and fusion growth charts. A log
        # axis here would need to be labelled as one, and mixing scales across
        # otherwise-identical growth charts invites misreading.
        figure.update_yaxes(title="Growth of $1", tickprefix="$", tickformat=".2f")
    else:
        figure.add_trace(
            go.Scatter(
                x=series["date"],
                y=series["drawdown"] * 100,
                mode="lines",
                name=fund_name,
                line={"color": colour, "width": 1.8},
                fill="tozeroy",
                fillcolor="rgba(74, 85, 104, 0.16)",
                hovertemplate="%{x|%d %b %Y}<br>Drawdown: %{y:.2f}%<extra></extra>",
            )
        )
        _apply_layout(figure, f"Drawdown · {fund_name}")
        figure.update_yaxes(title="Drawdown (%)", ticksuffix="%")
        figure.add_hline(y=0, line_color=SENTIMENT_ZERO_COLOR, line_width=1)
    figure.update_xaxes(title="Date")
    return figure


def make_allocation_chart(timeline: pd.DataFrame) -> go.Figure:
    figure = go.Figure(
        go.Scatter(
            x=timeline["date"],
            y=timeline["portfolio_value"],
            mode="lines",
            name="Portfolio value",
            line={"color": ALLOCATION_COLOR, "width": 2.2},
            hovertemplate="%{x|%d %b %Y}<br>Portfolio Value: $%{y:,.2f}<extra></extra>",
        )
    )
    _apply_layout(figure, "Illustrative portfolio value")
    figure.update_xaxes(title="Date")
    figure.update_yaxes(title="Portfolio Value ($)", tickprefix="$", tickformat=",.0f")
    return figure


def make_sector_sentiment_chart(sector_rows: pd.DataFrame, sector: str) -> go.Figure:
    rolling = rolling_mean_of_valid_observations(
        sector_rows.set_index("date")["extended_z_expanding"]
    )
    hover_extra = sector_rows[["headline_count", "active_ticker_count"]].to_numpy()

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=sector_rows["date"],
            y=sector_rows["extended_z_expanding"],
            mode="lines",
            name="Daily z-score",
            line={"color": SENTIMENT_DAILY_COLOR, "width": 1},
            connectgaps=False,
            customdata=hover_extra,
            hovertemplate=(
                "Daily z: %{y:.2f}<br>"
                "Headline Count: %{customdata[0]}<br>"
                "Active Tickers: %{customdata[1]}"
                "<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=rolling.index,
            y=rolling.to_numpy(),
            mode="lines",
            name=f"{ROLLING_WINDOW}-observation rolling mean",
            line={"color": SENTIMENT_ROLLING_COLOR, "width": 2.4},
            connectgaps=False,
            hovertemplate="Rolling z: %{y:.2f}<extra></extra>",
        )
    )
    figure.add_hline(y=0, line_color=SENTIMENT_ZERO_COLOR, line_width=1, line_dash="dot")
    for threshold in (1.5, -1.5):
        figure.add_hline(
            y=threshold, line_color=SENTIMENT_THRESHOLD_COLOR, line_width=1, line_dash="dash"
        )
    _apply_layout(figure, f"Extended sentiment index · {sector}")
    figure.update_xaxes(title="Date")
    figure.update_yaxes(title="Standardised sentiment (expanding z-score)")
    return figure


def make_fusion_chart(fusion_returns: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    for variant, style in FUSION_STYLES.items():
        rows = fusion_returns.loc[fusion_returns["variant"] == variant].sort_values("date")
        if rows.empty:
            continue
        figure.add_trace(
            go.Scatter(
                x=rows["date"],
                y=rows["growth_of_1"],
                mode="lines",
                name=variant,
                line={"color": style["color"], "dash": style["dash"], "width": style["width"]},
                hovertemplate=f"{variant}: $%{{y:.3f}}<extra></extra>",
            )
        )
    figure.add_hline(y=1.0, line_color=SENTIMENT_ZERO_COLOR, line_width=1, line_dash="dot")
    _apply_layout(figure, "Growth of $1 · base fund vs sentiment tilt")
    figure.update_xaxes(title="Date")
    figure.update_yaxes(title="Growth of $1", tickprefix="$", tickformat=".2f")
    return figure


# ============================================================================
# Page: Fund Explorer
# ============================================================================


def render_fund_explorer() -> None:
    metrics = load_or_stop(load_performance_metrics)
    universe = summarise_fund_universe(metrics)

    st.title(APP_TITLE)
    st.write(
        "Compare systematic funds across return, risk, drawdown and current holdings."
    )
    st.caption(
        f"{universe['n_funds']} funds · {universe['n_families']} asset universes · "
        f"{universe['n_methods']} methods · "
        f"{universe['start'].year}–{universe['end'].year}"
    )
    st.info(
        "Historical backtest results are net of the project's transaction-cost "
        "assumption and do not guarantee future performance.",
        icon=":material/info:",
    )

    st.subheader("Compare funds")
    left, right = st.columns(2)
    families = ["All", *sorted(metrics["fund_family"].unique())]
    methods = ["All", *sorted(metrics["method"].unique())]
    with left:
        family_choice = st.selectbox("Fund Family", families)
    with right:
        method_choice = st.selectbox(
            "Portfolio Method", methods,
            format_func=lambda value: value if value == "All" else method_label(value),
        )

    filtered = metrics
    if family_choice != "All":
        filtered = filtered.loc[filtered["fund_family"] == family_choice]
    if method_choice != "All":
        filtered = filtered.loc[filtered["method"] == method_choice]

    if filtered.empty:
        st.warning("No funds match this combination of filters. Try widening the selection.")
    else:
        st.plotly_chart(make_risk_return_chart(filtered), width="stretch")
        comparison = build_fund_comparison_table(filtered)
        st.dataframe(
            comparison, width="stretch", hide_index=True,
            column_config=fund_comparison_column_config(),
            height=table_height(comparison),
        )

    st.divider()
    st.subheader("Fund fact sheet")

    fund_names = list(metrics["fund_name"])
    default_index = (
        fund_names.index(DEFAULT_FACT_SHEET_FUND)
        if DEFAULT_FACT_SHEET_FUND in fund_names
        else 0
    )
    selected_fund = st.selectbox("Select a fund", fund_names, index=default_index)
    fund_row = metrics.loc[metrics["fund_name"] == selected_fund].iloc[0]
    colour = FAMILY_COLORS.get(fund_row["fund_family"], "#4A5568")

    top_row = st.columns(2)
    bottom_row = st.columns(2)
    top_row[0].metric("Annual Return", format_percent(fund_row["annual_return"]))
    top_row[1].metric("Annual Volatility", format_percent(fund_row["annual_volatility"]))
    bottom_row[0].metric("Sharpe Ratio", format_ratio(fund_row["sharpe_ratio"]))
    bottom_row[1].metric("Maximum Drawdown", format_percent(fund_row["max_drawdown"]))

    chart_mode = st.radio(
        "Chart", ["Growth of $1", "Drawdown"], horizontal=True, key="fact_sheet_chart"
    )
    returns = load_or_stop(load_fund_returns)
    series = returns.loc[returns["fund_name"] == selected_fund].sort_values("date")
    if series.empty:
        st.warning("No daily history is available for this fund.")
    else:
        # Only the selected chart is built, so a page view never renders both.
        st.plotly_chart(
            make_fund_time_series_chart(series, selected_fund, colour, chart_mode),
            width="stretch",
        )
        st.caption(
            "Growth of $1 is net of the project's transaction-cost assumption. "
            f"Out-of-sample window: {format_date(series['date'].min())} to "
            f"{format_date(series['date'].max())}."
        )

    st.markdown("**Current holdings**")
    holdings = load_or_stop(load_current_holdings)
    fund_holdings = holdings.loc[holdings["fund_name"] == selected_fund]
    if fund_holdings.empty:
        st.info("No current holdings are recorded for this fund.")
    else:
        ranked = fund_holdings.sort_values("target_weight", ascending=False)
        # Two columns rather than four: at a narrow window a quarter-width
        # metric truncates both the label and a formatted date.
        info_top = st.columns(2)
        info_bottom = st.columns(2)
        info_top[0].metric("Latest Formation Date", format_date(ranked["formation_date"].max()))
        info_top[1].metric("Effective Date", format_date(ranked["effective_date"].max()))
        info_bottom[0].metric("Active Holdings", f"{len(ranked)}")
        info_bottom[1].metric(
            "Largest Target Weight", format_percent(ranked["target_weight"].max())
        )
        top_holdings = build_holdings_table(ranked)
        st.dataframe(
            top_holdings, width="stretch", hide_index=True,
            column_config=holdings_column_config(),
            height=table_height(top_holdings, max_height=420),
        )

    with st.expander("How this fund is constructed"):
        st.markdown(_method_description(fund_row["method"]))
        st.markdown(
            "- Weights are formed monthly on a walk-forward basis and first held on the "
            "next trading day, so a formation-date signal never affects that day's return.\n"
            "- The fund is long-only and fully invested.\n"
            "- Returns are net of the project's transaction-cost assumption.\n"
            "- All results shown are out-of-sample; historical results are not a guarantee "
            "of future performance."
        )


def _method_description(method: str) -> str:
    descriptions = {
        "equal_weight": (
            "**Equal Weight** spreads capital evenly across every eligible asset. "
            "It does not estimate expected returns or a covariance matrix, reducing "
            "parameter-estimation risk in the portfolio weights."
        ),
        "minimum_variance": (
            "**Minimum Variance** solves for the long-only weights with the lowest "
            "estimated portfolio variance, using the covariance of past returns."
        ),
        "maximum_sharpe": (
            "**Maximum Sharpe** solves for the long-only weights with the highest "
            "estimated return-to-risk ratio (risk-free rate assumed zero). It tends to "
            "concentrate into fewer names than the other methods."
        ),
        "hrp": (
            "**HRP (Hierarchical Risk Parity)** clusters assets by correlation, then "
            "allocates risk down the resulting hierarchy. It avoids inverting the "
            "covariance matrix, so it is less sensitive to estimation noise."
        ),
    }
    return descriptions.get(method, f"**{method_label(method)}**.")


# ============================================================================
# Page: Allocation Lab
# ============================================================================


def render_allocation_lab() -> None:
    st.title("Allocation Lab")
    st.write("Combine 2–4 funds into an illustrative buy-and-hold allocation.")

    returns = load_or_stop(load_fund_returns)
    fund_names = sorted(returns["fund_name"].unique())
    defaults = [name for name in DEFAULT_ALLOCATION_FUNDS if name in fund_names]
    if len(defaults) < 2:
        defaults = fund_names[: min(3, len(fund_names))]

    selected = st.multiselect(
        "Select 2 to 4 funds", fund_names, default=defaults, max_selections=4
    )
    if len(selected) < 2:
        st.info("Select at least two funds to build an allocation.")
        return

    heading, reset = st.columns([4, 1])
    heading.markdown("**Set your allocation**")
    # Rendered before the inputs so a click can rewrite their state in the
    # same run; changing a widget's state after it exists is not allowed.
    reset_clicked = reset.button("Reset to equal weights", width="stretch")

    sync_allocation_weight_state(selected, force_reset=reset_clicked)

    columns = st.columns(len(selected))
    allocations_pct: dict[str, float] = {}
    for column, fund in zip(columns, selected):
        with column:
            allocations_pct[fund] = st.number_input(
                fund, min_value=0.0, max_value=100.0,
                step=1.0, format="%.2f", key=allocation_state_key(fund),
            )

    total_pct = sum(allocations_pct.values())
    investment = st.number_input(
        "Initial Investment ($)", min_value=1.0, value=10_000.0, step=500.0, format="%.2f"
    )

    if abs(total_pct - 100.0) > 1e-6:
        st.error(
            f"Allocations currently add up to {total_pct:.2f}%. "
            "Adjust them to total exactly 100% to see the result."
        )
        return

    try:
        result = calculate_buy_and_hold_allocation(
            returns,
            {fund: pct / 100.0 for fund, pct in allocations_pct.items()},
            investment,
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    metric_row = st.columns(4)
    metric_row[0].metric("Initial Investment", format_dollars(result["initial_investment"]))
    metric_row[1].metric("Ending Value", format_dollars(result["ending_value"]))
    metric_row[2].metric("Cumulative Return", format_percent(result["cumulative_return"]))
    metric_row[3].metric("Maximum Drawdown", format_percent(result["max_drawdown"]))

    st.plotly_chart(make_allocation_chart(result["timeline"]), width="stretch")
    st.caption(
        f"Evaluated from {format_date(result['common_start'])} to "
        f"{format_date(result['end_date'])} - the period over which every selected fund "
        "has a published value. Funds on different trading calendars keep their last "
        "published value on days they do not trade."
    )

    allocation_table = build_allocation_table(allocations_pct)
    st.dataframe(
        allocation_table, width="stretch", hide_index=True,
        column_config=allocation_column_config(),
        height=table_height(allocation_table),
    )

    st.caption(
        "Illustrative allocation only. The result assumes an initial allocation followed "
        "by buy-and-hold, does not execute trades, and is not personalised financial advice."
    )


# ============================================================================
# Page: Sentiment & Fusion
# ============================================================================


def render_sector_sentiment() -> None:
    sentiment = load_or_stop(load_sector_sentiment)
    sectors = sorted(sentiment["sector"].unique())
    default_index = sectors.index(DEFAULT_SECTOR) if DEFAULT_SECTOR in sectors else 0
    sector = st.selectbox("Select an equity sector", sectors, index=default_index)

    sector_rows = sentiment.loc[sentiment["sector"] == sector].sort_values("date")
    latest_trading_date = sector_rows["date"].max()
    observed = sector_rows.dropna(subset=["extended_z_expanding"])

    if observed.empty:
        st.warning(f"No sentiment observation is available for {sector}.")
        return

    latest_observation = observed.iloc[-1]
    latest_available_date = pd.Timestamp(latest_observation["date"])

    if latest_available_date != pd.Timestamp(latest_trading_date):
        st.warning(
            "No sentiment observation on the latest trading day "
            f"({format_date(latest_trading_date)}). "
            f"Latest available reading: {format_date(latest_available_date)}."
        )

    # Two metrics per row: at three columns the long z-score label is
    # truncated, and the two date labels are easy to confuse when clipped.
    date_row = st.columns(2)
    date_row[0].metric("Latest Trading Date", format_date(latest_trading_date))
    date_row[1].metric("Latest Available Sentiment Date", format_date(latest_available_date))

    signal_row = st.columns(2)
    signal_row[0].metric(
        "Extended sentiment z-score vs prior history",
        format_ratio(latest_observation["extended_z_expanding"]),
    )
    signal_row[1].metric(
        "Latest Band", str(latest_observation["extended_band"]).replace("_", " ").title()
    )

    detail_row = st.columns(2)
    detail_row[0].metric(
        "Headlines That Day", f"{int(latest_observation['headline_count'])}"
    )
    detail_row[1].metric(
        "Active Ticker Coverage", format_percent(latest_observation["coverage_ratio"], 1)
    )

    st.plotly_chart(make_sector_sentiment_chart(sector_rows, sector), width="stretch")

    observed_share = len(observed) / len(sector_rows) if len(sector_rows) else float("nan")
    st.caption(
        f"{sector} has a sentiment reading on {len(observed):,} of "
        f"{len(sector_rows):,} trading days ({format_percent(observed_share, 1)}) between "
        f"{format_date(sector_rows['date'].min())} and {format_date(latest_trading_date)}. "
        "A positive z-score means sentiment is above this sector's own prior-history "
        "benchmark and a negative z-score means below it; zero means close to that "
        "benchmark, which is not the same as no information. Gaps are days with no "
        "usable reading and are left blank rather than filled. The rolling line is a "
        "display-only smoother and is not used by the index, the classification or the tilt."
    )

    with st.expander("Why Extended FinVADER?"):
        st.markdown(
            "- The production signal uses Extended FinVADER: the general-purpose lexicon "
            "plus a frozen 30-term finance lexicon built for this project.\n"
            "- The baseline-versus-extended validation is documented separately in the "
            "model-validation exhibit; this app shows the final signal only.\n"
            "- Headline sentiment is a noisy proxy for news content and is not a trading "
            "guarantee."
        )


def render_sentiment_tilt() -> None:
    st.markdown(
        "Extended sentiment is applied as a lagged monthly tilt to the "
        "**Equity-Only Minimum Variance** fund."
    )
    st.latex(r"\tilde{w}_{i,t} = w^{base}_{i,t}\,\bigl(1 + \lambda\, z_{i,t-1}\bigr)")
    st.markdown(
        "- **Base** uses $\\lambda = 0$, **Momentum** uses $\\lambda = +1$ and "
        "**Contrarian** uses $\\lambda = -1$.\n"
        "- A missing signal gives a multiplier of exactly 1, so the weight is left "
        "untouched - that records missing information, not neutral sentiment.\n"
        "- Negative tilted weights are clipped to zero and the weights are renormalised, "
        "keeping the fund long-only and fully invested.\n"
        "- The signal is lagged by at least one equity trading day, so a target weight "
        "only ever uses sentiment that was already observable.\n"
        "- The two tilt directions were fixed before the results were evaluated; neither "
        "was selected afterwards."
    )

    fusion_returns = load_or_stop(load_fusion_returns)
    st.plotly_chart(make_fusion_chart(fusion_returns), width="stretch")

    summary = load_or_stop(load_fusion_summary)
    table = build_fusion_comparison_table(summary)
    st.dataframe(
        table, width="stretch", hide_index=True,
        column_config=fusion_comparison_column_config(),
        height=table_height(table),
    )

    for variant in summary["variant"]:
        if variant == FUSION_BASE_VARIANT:
            continue
        st.markdown(describe_fusion_variant(summary, variant))

    coverage = load_or_stop(load_fusion_coverage)
    facts = summarise_fusion_coverage(coverage)
    st.caption(
        f"Across {facts['n_formations']} monthly formation dates between "
        f"{format_date(facts['start'])} and {format_date(facts['end'])}, an extended "
        f"sentiment reading was available for {format_percent(facts['mean_coverage'], 1)} "
        f"of stocks on average, and {facts['inactive_count']} formation date(s) had no "
        "usable signal for any held position. The ticker-level expanding z-score needs "
        f"{EXPANDING_MIN_PERIODS} prior non-missing observations before it produces a "
        "value, unavailable signals are never carried forward, and a formation date "
        "with no usable signal is left untilted."
    )


def render_sentiment_and_fusion() -> None:
    st.title("Sentiment & Fusion")
    st.write(
        "Explore sector news sentiment and how a lagged sentiment tilt changes the base fund."
    )
    view = st.radio("View", ["Sector Sentiment", "Sentiment Tilt"], horizontal=True)
    # Only the selected sub-view is built.
    if view == "Sector Sentiment":
        render_sector_sentiment()
    else:
        render_sentiment_tilt()


# ============================================================================
# Main navigation
# ============================================================================


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide")

    st.sidebar.title(APP_TITLE)
    st.sidebar.write(APP_TAGLINE)
    page = st.sidebar.radio(
        "Navigate", ["Fund Explorer", "Allocation Lab", "Sentiment & Fusion"]
    )
    st.sidebar.divider()
    st.sidebar.caption(APP_DISCLAIMER)

    # Explicit dispatch: the renderers for unselected pages are never called,
    # so their CSVs are never read and their charts are never built.
    if page == "Fund Explorer":
        render_fund_explorer()
    elif page == "Allocation Lab":
        render_allocation_lab()
    else:
        render_sentiment_and_fusion()


if __name__ == "__main__":
    main()
