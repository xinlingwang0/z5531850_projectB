"""Reproduce RiskBridge Funds Part B results.

Run from the project root:

    python scripts/run_part_b.py
    python scripts/run_part_b.py --sentiment-candidates-only
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from collections.abc import Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from cycler import cycler
from matplotlib.lines import Line2D

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src import data_access, etl, features, fusion, portfolios, sentiment  # noqa: E402

# ============================================================================
# Shared paths (used by every Station 3 build below)
# ============================================================================

TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
DATA = ROOT / "results" / "data"


def ensure_dirs() -> None:
    for path in (TABLES, FIGURES, DATA):
        path.mkdir(parents=True, exist_ok=True)


def save_csv(df: pd.DataFrame, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"wrote {path.relative_to(ROOT)}")


# ============================================================================
# Station 3 - Portfolio construction (funds): backtest, tables, figures
# ============================================================================

# Fund-family and method identity shared by every Station 3 figure below, so the
# same colour always means the same thing across every exhibit (a coherent
# RiskBridge design language, matching the visual style already used in Part
# A's scripts/run_part_a.py).
FUND_FAMILIES = ("Equity-Only", "Crypto-Only", "Combined")
METHOD_ORDER = tuple(portfolios.METHODS)
FAMILY_COLORS = {
    "Equity-Only": "#1F3A5F",  # navy
    "Crypto-Only": "#C99700",  # gold
    "Combined": "#007C89",  # teal
}
METHOD_COLORS = {
    "equal_weight": "#707070",  # neutral grey - naive benchmark, deliberately not blue
    "minimum_variance": "#2E86C1",  # bright blue - was "#1F3A5F", identical to
    # FAMILY_COLORS["Equity-Only"] and too close to the old equal_weight grey-blue
    "maximum_sharpe": "#B23A48",  # crimson
    "hrp": "#2E7D32",  # forest
}
METHOD_MARKERS = {
    "equal_weight": "o",
    "minimum_variance": "s",
    "maximum_sharpe": "^",
    "hrp": "D",
}
FUND_SOURCE = (
    "FINS3645 equity and crypto price bundles; RiskBridge walk-forward "
    "out-of-sample fund backtest (Part B, Station 3)"
)

# Same palette and A4/Word-ready sizing as fintools.figures.theme.theme_rc
# (profile="word_a4", style="fins"), copied as literal values rather than
# imported. This project folder is graded and zipped on its own (see
# PROJECT_BRIEF.md section 7: "open only your own project folder"), so it must
# run end to end without depending on the top-level fins-agent repo.
THEME_RC = {
    "axes.edgecolor": "#2F3337",
    "axes.grid": False,
    "axes.labelcolor": "#1F2933",
    "axes.labelsize": 11,
    "axes.linewidth": 0.8,
    "axes.prop_cycle": cycler(color=list(FAMILY_COLORS.values())),
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.titlecolor": "#111827",
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "figure.constrained_layout.use": False,
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "font.family": "DejaVu Sans",
    "grid.alpha": 0.55,
    "grid.color": "#D8DDE6",
    "grid.linewidth": 0.6,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "lines.linewidth": 1.8,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "savefig.pad_inches": 0.04,
    "xtick.color": "#4B5563",
    "xtick.labelsize": 10,
    "ytick.color": "#4B5563",
    "ytick.labelsize": 10,
}


def apply_theme() -> None:
    mpl.rcParams.update(THEME_RC)


def add_figure_heading(
    title: str, note: str, *, fig: mpl.figure.Figure | None = None
) -> tuple[mpl.text.Text, mpl.text.Text]:
    """Add a consistent bold title and regular-weight explanatory note."""
    fig = plt.gcf() if fig is None else fig
    title_size = 12.0
    note_size = 10.0
    title_y = 0.985
    figure_height = float(fig.get_size_inches()[1])

    title_artist = fig.suptitle(
        title,
        x=0.5,
        y=title_y,
        ha="center",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color="#111827",
    )
    title_height = title_size * 1.25 / (72.0 * figure_height)
    interline_gap = 2.0 / (72.0 * figure_height)
    note_y = title_y - title_height - interline_gap
    note_artist = fig.text(
        0.5,
        note_y,
        note,
        ha="center",
        va="top",
        fontsize=note_size,
        fontweight="normal",
        linespacing=1.25,
        color="#111827",
    )

    note_lines = note.count("\n") + 1
    note_height = note_size * 1.25 * note_lines / (72.0 * figure_height)
    bottom_gap = 4.0 / (72.0 * figure_height)
    fig._riskbridge_heading_bottom = note_y - note_height - bottom_gap
    return title_artist, note_artist


def add_figure_footer(source: str, sample: str) -> None:
    """Add a publication-style source/sample footer inside the saved figure."""
    footer = f"Source: {source} | Sample period: {sample}"
    fig = plt.gcf()
    fig.text(0.01, 0.015, footer, ha="left", va="bottom", fontsize=7.5, color="dimgray")


def finalise_figure(
    path: pathlib.Path, source: str, sample: str, *, top: float = 0.95
) -> pathlib.Path:
    """Apply the common footer, layout and save settings used by every figure below."""
    add_figure_footer(source, sample)
    fig = plt.gcf()
    heading_bottom = float(getattr(fig, "_riskbridge_heading_bottom", top))
    plt.tight_layout(rect=[0, 0.07, 1, min(top, heading_bottom)])
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def _date_range(df: pd.DataFrame, date_col: str) -> str:
    dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if dates.empty:
        return "n/a"
    return f"{dates.min().date()} to {dates.max().date()}"


def _metrics_sample_range(performance_metrics: pd.DataFrame) -> str:
    start = pd.to_datetime(performance_metrics["oos_start"], errors="coerce").min()
    end = pd.to_datetime(performance_metrics["oos_end"], errors="coerce").max()
    if pd.isna(start) or pd.isna(end):
        return "n/a"
    return f"{start.date()} to {end.date()}"


def plot_fund_growth(fund_returns: pd.DataFrame) -> pathlib.Path:
    """Figure 1: out-of-sample growth of $1 by fund family and method."""
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.0), sharey=False)
    for ax, family in zip(axes, FUND_FAMILIES):
        panel = fund_returns.loc[fund_returns["fund_family"] == family]
        endpoints = []
        for method in METHOD_ORDER:
            series = panel.loc[panel["method"] == method].sort_values("date")
            if series.empty:
                continue
            ax.plot(
                series["date"],
                series["growth_of_1"],
                label=portfolios.METHOD_LABELS[method],
                color=METHOD_COLORS[method],
                linewidth=1.6,
            )
            endpoints.append(
                (method, series["date"].iloc[-1], float(series["growth_of_1"].iloc[-1]))
            )
        ax.axhline(1.0, color="#4B5563", linewidth=0.6, linestyle=":")
        # Log scale: equal visual distance means equal proportional change, so
        # Crypto-Only's much wider swings do not force Equity-Only and Combined
        # into two flat, unreadable lines under a shared or linear axis.
        ax.set_yscale("log")
        # Centre each panel's own ylim on 1.0 in log space (symmetric ratio
        # above and below break-even), so the $1 reference line sits at the
        # same relative height in every panel even though the three panels
        # keep independent, differently-scaled y-axes.
        ratio = max(panel["growth_of_1"].max() / 1.0, 1.0 / panel["growth_of_1"].min())
        padded_ratio = ratio * 1.15
        ax.set_ylim(1.0 / padded_ratio, 1.0 * padded_ratio)
        # A padded ratio close to 1.0 (Equity-Only, Combined) can leave every
        # standard log-decade tick (...0.1, 1, 10...) outside this narrow a
        # range except "1", so pick 5 ticks explicitly spaced in log space -
        # symmetric around 1.0 like the ylim itself - and label them as plain
        # dollar-growth numbers rather than power-of-ten notation.
        log_half_span = np.log10(padded_ratio)
        tick_values = 10 ** np.linspace(-log_half_span, log_half_span, 5)
        ax.set_yticks(tick_values)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.set_title(family, fontsize=10.5)
        ax.set_xlabel("Date")
        ax.tick_params(axis="x", labelrotation=30)

        # End-of-sample value labels, stacked in log space so lines finishing
        # close together get equal visual spacing instead of overlapping text.
        # A small dot marks the true data point; the printed "$X.XX" is always
        # the exact value, only its label position is nudged apart.
        if endpoints:
            endpoints.sort(key=lambda item: item[2])
            y_lo, y_hi = ax.get_ylim()
            min_gap = (np.log10(y_hi) - np.log10(y_lo)) * 0.10
            stacked_log = []
            for _, _, y_value in endpoints:
                candidate = np.log10(y_value)
                if stacked_log and candidate < stacked_log[-1] + min_gap:
                    candidate = stacked_log[-1] + min_gap
                stacked_log.append(candidate)
            # Reserve a slim, unlabeled area after the final observation for
            # direct endpoint labels. Major ticks stop at the real sample end,
            # so the padding cannot be mistaken for additional return data.
            sample_start = panel["date"].min()
            sample_end = panel["date"].max()
            date_span = sample_end - sample_start
            label_x = sample_end + date_span * 0.018
            ax.set_xlim(left=sample_start, right=sample_end + date_span * 0.10)
            tick_dates = pd.date_range(
                start=pd.Timestamp(year=sample_start.year, month=1, day=1),
                end=sample_end,
                freq="4MS",
            )
            tick_dates = tick_dates[tick_dates >= sample_start]
            ax.set_xticks(tick_dates)
            ax.set_xticklabels(tick_dates.strftime("%Y-%m"))
            for (method, x_last, y_value), label_log in zip(endpoints, stacked_log):
                ax.scatter(
                    [x_last], [y_value], color=METHOD_COLORS[method], s=16,
                    zorder=4, edgecolor="white", linewidth=0.6,
                    clip_on=False,
                )
                ax.annotate(
                    f"${y_value:.2f}", xy=(x_last, y_value),
                    xytext=(label_x, 10 ** label_log), textcoords="data",
                    color=METHOD_COLORS[method], fontsize=8.5, fontweight="bold",
                    va="center", ha="left",
                    arrowprops={
                        "arrowstyle": "-",
                        "color": METHOD_COLORS[method],
                        "linewidth": 0.7,
                        "alpha": 0.75,
                        "shrinkA": 1.5,
                        "shrinkB": 2.0,
                    },
                    zorder=6,
                )
    axes[0].set_ylabel("Growth of $1")
    axes[0].legend(loc="upper left", fontsize=8)
    add_figure_heading(
        "Figure 1. Out-of-sample growth of $1 by fund family and method",
        "Walk-forward monthly rebalancing; each formation date uses only information available up to that date.\n"
        "Note: each panel's y-axis is scaled independently (log scale) - compare methods within a panel, not heights across panels.",
        fig=fig,
    )
    path = FIGURES / "figure_1_fund_growth_of_1.png"
    return finalise_figure(path, FUND_SOURCE, _date_range(fund_returns, "date"), top=0.80)


def plot_flagship_drawdown(fund_returns: pd.DataFrame) -> pathlib.Path:
    """Figure 2: drawdown for the flagship Combined Maximum Sharpe fund vs. the equal-weight benchmark."""
    panel = fund_returns.loc[fund_returns["fund_family"] == "Combined"]
    flagship = panel.loc[panel["method"] == "maximum_sharpe"].sort_values("date")
    benchmark = panel.loc[panel["method"] == "equal_weight"].sort_values("date")

    plt.figure(figsize=(9.0, 5.0))
    plt.fill_between(
        flagship["date"], flagship["drawdown"] * 100, 0,
        color=METHOD_COLORS["maximum_sharpe"], alpha=0.28,
    )
    plt.plot(
        flagship["date"], flagship["drawdown"] * 100,
        color=METHOD_COLORS["maximum_sharpe"], linewidth=1.5, label="Combined Maximum Sharpe",
    )
    plt.plot(
        benchmark["date"], benchmark["drawdown"] * 100,
        color=METHOD_COLORS["equal_weight"], linewidth=1.2, linestyle="--",
        label="Combined Equal Weight (benchmark)",
    )
    plt.axhline(0, color="#262A33", linewidth=0.6)
    add_figure_heading(
        "Figure 2. Drawdown: Combined Maximum Sharpe vs. the equal-weight benchmark",
        "Peak-to-trough decline in growth of $1; the flagship optimised fund is checked against the simplest naive benchmark.",
    )
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")
    plt.legend(loc="lower left", fontsize=8)
    path = FIGURES / "figure_2_flagship_drawdown.png"
    return finalise_figure(path, FUND_SOURCE, _date_range(panel, "date"))


def plot_combined_weights_by_asset_class(
    fund_weights: pd.DataFrame,
    equity_tickers: set[str],
    crypto_tickers: set[str],
) -> pathlib.Path:
    """Figure 3: Combined-fund equity/crypto weight split over time, across all four methods.

    Fifty equity and ten crypto tickers make a per-ticker stacked area chart
    unreadable, so weights are grouped to the asset-class level. This is also
    the economically interesting comparison for a combined fund: how much
    crypto exposure each optimiser actually carries, and whether that changes
    over time. All four construction methods are shown (not a subset), so the
    required "weights over time across methods" exhibit is not missing HRP and
    Equal Weight.
    """
    panel = fund_weights.loc[fund_weights["fund_family"] == "Combined"].copy()
    panel["asset_class"] = np.where(panel["ticker"].isin(crypto_tickers), "Crypto", "Equity")

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0), sharey=True, sharex=True)
    axes_flat = axes.flatten()
    for idx, method in enumerate(METHOD_ORDER):
        ax = axes_flat[idx]
        sub = panel.loc[panel["method"] == method]
        grouped = (
            sub.groupby(["effective_date", "asset_class"])["target_weight"]
            .sum()
            .unstack("asset_class")
            .reindex(columns=["Equity", "Crypto"], fill_value=0.0)
            .sort_index()
        )
        ax.stackplot(
            grouped.index,
            grouped["Equity"] * 100,
            grouped["Crypto"] * 100,
            labels=["Equity", "Crypto"],
            colors=[FAMILY_COLORS["Equity-Only"], FAMILY_COLORS["Crypto-Only"]],
            alpha=0.85,
        )
        ax.set_title(portfolios.METHOD_LABELS[method], fontsize=10.5)
        ax.tick_params(axis="x", labelrotation=30)
        ax.set_ylim(0, 100)
        if idx % 2 == 0:
            ax.set_ylabel("Target weight (%)")
        if idx >= 2:
            ax.set_xlabel("Effective date")
    axes_flat[1].legend(loc="upper right", fontsize=8)
    add_figure_heading(
        "Figure 3. Combined-fund equity/crypto split over time, across methods",
        "Target weights from each monthly formation date, grouped to the asset-class level for readability.",
        fig=fig,
    )
    path = FIGURES / "figure_3_combined_weights_by_asset_class.png"
    return finalise_figure(
        path, FUND_SOURCE, _date_range(panel, "effective_date"), top=0.90
    )


def plot_sharpe_barplot(performance_metrics: pd.DataFrame) -> pathlib.Path:
    """Figure 4: annualised Sharpe ratio (rf = 0) across funds and methods.

    Reads sharpe_ratio directly from performance_metrics (the same DataFrame
    run_full_build() saves to results/tables/performance_metrics.csv) rather
    than recomputing it, so the bar heights and the table can never disagree.
    """
    df = performance_metrics.set_index(["fund_family", "method"])["sharpe_ratio"]

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    x = np.arange(len(FUND_FAMILIES))
    n_methods = len(METHOD_ORDER)
    width = 0.8 / n_methods
    offsets = (np.arange(n_methods) - (n_methods - 1) / 2) * width

    for offset, method in zip(offsets, METHOD_ORDER):
        heights = [df.get((family, method), np.nan) for family in FUND_FAMILIES]
        bars = ax.bar(
            x + offset, heights, width,
            label=portfolios.METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            edgecolor="#262A33",
            linewidth=0.5,
        )
        for bar, height in zip(bars, heights):
            if not np.isfinite(height):
                continue
            va = "bottom" if height >= 0 else "top"
            offset_pts = 3 if height >= 0 else -3
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, offset_pts),
                textcoords="offset points",
                ha="center", va=va, fontsize=7.5, color="#1F2933",
            )

    ax.axhline(0, color="#262A33", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(FUND_FAMILIES)
    ax.set_ylabel("Sharpe ratio (annualised, rf = 0)")
    add_figure_heading(
        "Figure 4. Annualised Sharpe ratio across funds and methods",
        "Sharpe ratios from the walk-forward out-of-sample backtest, grouped by fund family; risk-free rate assumed zero.",
        fig=fig,
    )
    # Extra headroom above the tallest bar so its value label never collides
    # with the "upper left" method legend, regardless of which bar is tallest.
    finite_values = df.dropna()
    if not finite_values.empty:
        y_max = float(finite_values.max())
        y_min = min(0.0, float(finite_values.min()))
        pad = (y_max - y_min) * 0.05
        ax.set_ylim(y_min - pad, y_max + (y_max - y_min) * 0.35)
    ax.legend(
        title="Method", loc="upper left", fontsize=8, title_fontsize=8,
        frameon=True, facecolor="white", edgecolor="#D8DDE6", framealpha=0.94,
    )

    path = FIGURES / "figure_4_sharpe_barplot.png"
    return finalise_figure(path, FUND_SOURCE, _metrics_sample_range(performance_metrics))


def plot_backtest_diagnostics(diagnostics: pd.DataFrame) -> pathlib.Path:
    """Figure 5 (extra, beyond the brief's required list): turnover and solver-fallback rate by method.

    The brief warns (section 8) that optimisers on small daily-return
    covariances can silently stall, and that weights should be checked to
    differ across methods. ``portfolios.build_fund_universe`` already records
    turnover and fallback usage per rebalance, so this figure turns that
    sanity check into evidence rather than a claim.
    """
    df = diagnostics.copy()
    df["method_label"] = df["method"].map(portfolios.METHOD_LABELS)
    method_labels = [portfolios.METHOD_LABELS[m] for m in METHOD_ORDER]

    turnover = (
        df.dropna(subset=["turnover"])
        .groupby(["fund_family", "method_label"])["turnover"]
        .mean()
        .unstack("fund_family")
        .reindex(method_labels)
    )
    fallback_rate = (
        df.groupby(["fund_family", "method_label"])["fallback_used"]
        .mean()
        .unstack("fund_family")
        .reindex(method_labels)
        * 100
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    x = np.arange(len(method_labels))
    width = 0.25
    for i, family in enumerate(FUND_FAMILIES):
        if family not in turnover.columns:
            continue
        axes[0].bar(
            x + (i - 1) * width, turnover[family] * 100, width,
            label=family, color=FAMILY_COLORS[family],
        )
        axes[1].bar(
            x + (i - 1) * width, fallback_rate[family], width,
            label=family, color=FAMILY_COLORS[family],
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(method_labels, rotation=20, ha="right")
    axes[0].set_ylabel("Average monthly turnover (%)")
    axes[0].set_title("Turnover by method", fontsize=10.5)
    axes[0].legend(loc="best", fontsize=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(method_labels, rotation=20, ha="right")
    axes[1].set_ylabel("Rebalances using equal-weight fallback (%)")
    axes[1].set_title("Solver fallback rate by method", fontsize=10.5)
    max_fallback = float(np.nan_to_num(fallback_rate.to_numpy(dtype=float)).max())
    axes[1].set_ylim(0, max(1.0, max_fallback * 1.4))
    if max_fallback == 0.0:
        axes[1].text(
            0.5, 0.5, "0% for every fund family and method\n(the optimiser never needed its fallback)",
            transform=axes[1].transAxes, ha="center", va="center", fontsize=10, color="#2E7D32",
        )

    add_figure_heading(
        "Figure 5 (extra). Backtest robustness: turnover and solver fallback by method",
        "Checks that methods actually trade differently and that the optimizer rarely needs its equal-weight fallback.",
        fig=fig,
    )
    path = FIGURES / "figure_5_backtest_diagnostics.png"
    return finalise_figure(
        path, FUND_SOURCE, _date_range(diagnostics, "effective_date"), top=0.86
    )


def run_full_build() -> None:
    """Build the Part B fund universe: clean data, backtest funds, save results."""
    ensure_dirs()
    apply_theme()

    print("Loading provided data through src.data_access ...")
    equities_raw = data_access.load_equity_prices()
    crypto_raw = data_access.load_crypto_prices()
    print("raw shapes:", "equities", equities_raw.shape, "crypto", crypto_raw.shape)

    equities, _ = etl.clean_price_panel(equities_raw, "equity_prices", has_sector=True)
    crypto, _ = etl.clean_price_panel(
        crypto_raw, "crypto_prices", end_date="2023-12-31", has_sector=False
    )

    eq_returns = features.daily_returns(equities, asset_class="Equity")
    cr_returns = features.daily_returns(crypto, asset_class="Crypto")

    # Crypto-Only keeps the native 365-day calendar (its own annualisation
    # factor); Combined uses crypto returns already aligned to equity dates.
    equity_wide = features.wide_returns(eq_returns)
    crypto_native_wide = features.wide_returns(cr_returns)
    crypto_aligned_wide = features.align_crypto_to_equity_calendar(
        cr_returns, eq_returns["date"].drop_duplicates()
    )
    combined_wide = features.combined_returns_panel(eq_returns, crypto_aligned_wide)

    print("Running walk-forward out-of-sample fund backtests ...")
    fund_universe = portfolios.build_fund_universe(
        equity_wide, crypto_native_wide, combined_wide, methods=portfolios.METHODS
    )
    fund_returns = fund_universe["fund_returns"]
    fund_weights = fund_universe["fund_weights"]
    performance_metrics = fund_universe["performance_metrics"]
    diagnostics = fund_universe["diagnostics"]

    save_csv(fund_returns, DATA / "fund_returns.csv")
    save_csv(fund_weights, DATA / "fund_weights.csv")
    save_csv(performance_metrics, TABLES / "performance_metrics.csv")
    save_csv(diagnostics, TABLES / "fund_backtest_diagnostics.csv")

    # Fact-sheet "current holdings": the most recent formation date's target
    # weights per (fund_family, method), non-zero positions only. Computed per
    # fund group so a fund family on a different native calendar (Crypto-Only
    # runs 365-day formation dates, not the equity/combined 252-day ones) still
    # resolves its own latest rebalance instead of being filtered out by a
    # global max(formation_date) across every fund.
    current_holdings = pd.concat(
        [
            portfolios.extract_current_holdings(group)
            for _, group in fund_weights.groupby(["fund_family", "method"])
        ],
        ignore_index=True,
    )
    current_holdings = current_holdings.loc[
        current_holdings["target_weight"] > portfolios.WEIGHT_TOLERANCE
    ].reset_index(drop=True)
    save_csv(current_holdings, TABLES / "current_holdings.csv")

    equity_tickers = set(equity_wide.columns)
    crypto_tickers = set(crypto_native_wide.columns)

    figure_paths = [
        plot_fund_growth(fund_returns),
        plot_flagship_drawdown(fund_returns),
        plot_combined_weights_by_asset_class(fund_weights, equity_tickers, crypto_tickers),
        plot_sharpe_barplot(performance_metrics),
        plot_backtest_diagnostics(diagnostics),
    ]
    for path in figure_paths:
        print(f"wrote {path.relative_to(ROOT)}")

    print("\nFund backtests complete.")

    # The fusion step consumes the ticker-day sentiment produced here rather
    # than reloading headlines or scoring them a second time.
    sentiment_results = run_sentiment_build()
    run_fusion_build(
        sentiment_results["ticker_day"],
        equity_wide,
        equities[["ticker", "sector"]].drop_duplicates(),
        eq_returns["date"].drop_duplicates(),
    )


# ============================================================================
# Station 3 - Sentiment: 2021-2023 extended sentiment build (baseline vs
# extended FinVADER, ticker/sector-day aggregation, expanding standardisation,
# and the automated baseline-extended comparison). 2020 is discovery-only (see
# the candidate-term section below, which this does not touch) - every table
# and figure produced here is restricted to 2021-01-01..2023-12-31.
# ============================================================================

SENTIMENT_SOURCE = (
    "FINS3645 equity news-headline bundle; RiskBridge baseline vs Extended "
    "FinVADER, RiskBridge 30-term custom lexicon (Part B, Station 3, "
    "application period 2021-2023)"
)
EXTENDED_SENTIMENT_SOURCE = (
    "FINS3645 equity news-headline bundle; RiskBridge Extended FinVADER "
    "with the RiskBridge 30-term custom lexicon (Part B, Station 3; "
    "application period 2021–2023)."
)
SECTOR_INDEX_MIN_PERIODS = 60


def _rolling_mean_of_valid_observations(
    daily_series: pd.Series, *, window: int = 21, min_periods: int = 21
) -> pd.Series:
    """Rolling mean over the latest ``window`` VALID (non-missing) observations,
    reindexed back onto ``daily_series``'s original (date-ordered) index.

    Dropping NaN first means the rolling window advances by count of valid
    observations, not by calendar position, so it never reaches back through
    a no-news gap to manufacture a smoothed value out of stale history. Any
    index label that was missing in ``daily_series`` is absent from the
    dropna'd series, so ``reindex`` puts NaN back there - the result is
    missing on exactly the same dates as the input, with no fill, no
    forward/backward fill, and no interpolation.
    """
    valid = daily_series.dropna()
    rolling = valid.rolling(window=window, min_periods=min_periods).mean()
    return rolling.reindex(daily_series.index)


def plot_extended_sector_sentiment_index(sector_index: pd.DataFrame) -> pathlib.Path:
    """Figure 6: 2021-2023 standardised Extended FinVADER sector sentiment index.

    10-sector small multiples (5x2), all sharing one y-axis range. Thin line:
    daily extended_z_expanding. Bold line: a rolling mean of the latest 21
    VALID sentiment observations (see _rolling_mean_of_valid_observations),
    for display only - it does not replace the daily index or the expanding
    z-score anywhere else in this build, and it stays missing on exactly the
    dates the daily index is missing (no fill, no carry-forward). Blank
    stretches are genuine missing values (no headlines that sector-day, or
    still inside the expanding window's first 60 non-missing observations) -
    never a fabricated 0 or "neutral".
    """
    sentiment.assert_application_period_only(
        sector_index["date"], context="figure_6 sector_index input"
    )
    sectors = sorted(sector_index["sector"].unique())
    fig, axes = plt.subplots(5, 2, figsize=(13.0, 13.5), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    abs_max = sector_index["extended_z_expanding"].abs().max()
    y_limit = max(3.0, float(abs_max) * 1.1) if pd.notna(abs_max) else 3.0

    for ax, sector in zip(axes_flat, sectors):
        g = sector_index.loc[sector_index["sector"] == sector].sort_values("date")
        rolling = _rolling_mean_of_valid_observations(g["extended_z_expanding"])
        ax.plot(g["date"], g["extended_z_expanding"], color="#B8AEA7", linewidth=0.7, alpha=0.85)
        ax.plot(g["date"], rolling, color=FAMILY_COLORS["Combined"], linewidth=1.8)
        ax.axhline(0, color="#4B5563", linewidth=0.6, linestyle=":")
        ax.axhline(1.5, color="#D8DDE6", linewidth=0.7, linestyle="--")
        ax.axhline(-1.5, color="#D8DDE6", linewidth=0.7, linestyle="--")
        ax.set_ylim(-y_limit, y_limit)
        ax.set_title(sector, fontsize=9.5)
        ax.tick_params(axis="x", labelrotation=30, labelsize=7.5)
        ax.tick_params(axis="y", labelsize=7.5)
    for ax in axes_flat[len(sectors):]:
        ax.axis("off")

    fig.supylabel(
        "Standardised Extended FinVADER sentiment\n(prior-history expanding z-score)",
        fontsize=10,
    )
    add_figure_heading(
        "Figure 6. Extended FinVADER sector sentiment, 2021-2023",
        "Daily prior-history z-scores and 21-observation rolling means",
        fig=fig,
    )
    path = FIGURES / "figure_6_extended_sector_sentiment_index.png"
    return finalise_figure(path, SENTIMENT_SOURCE, _date_range(sector_index, "date"), top=0.94)


def plot_baseline_vs_extended_sentiment(
    sector_index: pd.DataFrame, sector_comparison: pd.DataFrame
) -> pathlib.Path:
    """Figure 7: baseline vs. extended standardised sentiment for the overall
    aggregate, the sector with the highest custom-term hit rate, and the
    sector with the lowest. Selection rule: highest/lowest
    custom_term_hit_rate in sector_sentiment_model_comparison_table's
    per-sector rows (the "Overall" row is excluded from the min/max pick).
    2021-2023 only.

    The "Overall aggregate" panel is built from
    ``sentiment.daily_overall_sentiment_aggregate`` - the exact same
    equal-sector-weight daily series used for the "Overall" row of
    results/tables/sector_sentiment_model_comparison.csv, so the table and
    this figure can never disagree about what "Overall" means.
    """
    sentiment.assert_application_period_only(
        sector_index["date"], context="figure_7 sector_index input"
    )
    per_sector = sector_comparison.loc[sector_comparison["scope"] != "Overall"]
    highest_sector = str(per_sector.loc[per_sector["custom_term_hit_rate"].idxmax(), "scope"])
    lowest_sector = str(per_sector.loc[per_sector["custom_term_hit_rate"].idxmin(), "scope"])

    aggregate = sentiment.daily_overall_sentiment_aggregate(sector_index)

    panels = [
        ("Overall aggregate\n(equal-weight mean across sectors)", aggregate),
        (
            f"{highest_sector}\n(highest custom-lexicon coverage)",
            sector_index.loc[sector_index["sector"] == highest_sector].sort_values("date"),
        ),
        (
            f"{lowest_sector}\n(lowest custom-lexicon coverage)",
            sector_index.loc[sector_index["sector"] == lowest_sector].sort_values("date"),
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.2), sharey=True)
    for ax, (label, g) in zip(axes, panels):
        ax.plot(
            g["date"], g["baseline_z_expanding"], color=METHOD_COLORS["equal_weight"],
            linewidth=1.2, label="Baseline FinVADER",
        )
        ax.plot(
            g["date"], g["extended_z_expanding"], color=METHOD_COLORS["maximum_sharpe"],
            linewidth=1.2, label="Extended FinVADER",
        )
        ax.axhline(0, color="#4B5563", linewidth=0.6, linestyle=":")
        ax.set_title(label, fontsize=9.5)
        ax.tick_params(axis="x", labelrotation=30)
    axes[0].set_ylabel("Standardised sentiment (expanding z-score)")
    axes[0].legend(loc="upper left", fontsize=8)

    add_figure_heading(
        "Figure 7. Baseline versus Extended FinVADER, 2021-2023",
        "Overall market sentiment and sectors with the highest and lowest "
        "custom-lexicon coverage",
        fig=fig,
    )
    path = FIGURES / "figure_7_baseline_vs_extended_sentiment.png"
    return finalise_figure(path, SENTIMENT_SOURCE, _date_range(sector_index, "date"), top=0.88)


# ----------------------------------------------------------------------------
# Report-only sentiment exhibits (Figures 9 and 10)
#
# Both read the already-built sector index in memory: no headline is reloaded,
# nothing is re-scored, and the index itself is not recomputed. Both show the
# Extended model only - the baseline-versus-extended comparison is Figure 7's
# job and is not repeated here.
# ----------------------------------------------------------------------------

MARKET_SCORE_NEUTRAL = 50.0
# Same two sentiment line colours Figure 6 already uses, named here so the
# market-wide chart cannot drift away from the sector small multiples.
SENTIMENT_DAILY_COLOR = "#B8AEA7"
SENTIMENT_ROLLING_COLOR = FAMILY_COLORS["Combined"]  # "#007C89"
SENTIMENT_POSITIVE_COLOR = FAMILY_COLORS["Combined"]  # "#007C89"
SENTIMENT_NEGATIVE_COLOR = METHOD_COLORS["maximum_sharpe"]  # "#B23A48"


def market_score_100(extended_compound: pd.Series) -> pd.Series:
    """Map a raw compound score in [-1, 1] onto a 0-100 display scale.

    Purely a linear relabelling of the compound score: -1 -> 0, 0 -> 50,
    +1 -> 100. It is not a probability, a percentile, or a return forecast,
    and it carries exactly the information the compound score already had.
    """
    return MARKET_SCORE_NEUTRAL * (1.0 + extended_compound)


def plot_market_wide_sentiment_index(sector_index: pd.DataFrame) -> pathlib.Path:
    """Figure 9: market-wide Extended FinVADER sentiment, level and anomaly.

    The market-wide series is the shared
    ``sentiment.daily_overall_sentiment_aggregate`` - the same equal-weight,
    across-sector daily aggregate used by the Overall row of the model
    comparison table and by Figure 7 - rather than a second Overall defined
    here. Each date averages only the sectors with a valid reading that day,
    with raw compound and expanding z aggregated under their own validity
    masks, and the z aggregate is never re-standardised.
    """
    sentiment.assert_application_period_only(
        sector_index["date"], context="figure_9 sector_index input"
    )
    aggregate = sentiment.daily_overall_sentiment_aggregate(sector_index)
    aggregate = aggregate.sort_values("date").reset_index(drop=True)

    dates = aggregate["date"]
    score = market_score_100(aggregate["extended_compound"])
    rolling_score = _rolling_mean_of_valid_observations(score)
    anomaly = aggregate["extended_z_expanding"]

    fig, axes = plt.subplots(
        2, 1, figsize=(13.0, 8.6), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
    )

    upper = axes[0]
    upper.plot(
        dates, score, color=SENTIMENT_DAILY_COLOR, linewidth=0.8, alpha=0.9,
        label="Daily market sentiment score",
    )
    upper.plot(
        dates, rolling_score, color=SENTIMENT_ROLLING_COLOR, linewidth=2.2,
        label="21-observation rolling mean (display only)",
    )
    upper.axhline(
        MARKET_SCORE_NEUTRAL, color="#4B5563", linewidth=1.0, linestyle="--",
        label="Lexicon-neutral level (raw compound = 0)",
    )
    upper.set_ylabel("Market sentiment score (0-100)")
    upper.legend(
        loc="upper left", fontsize=8.5, frameon=True, facecolor="white",
        edgecolor="#D8DDE6", framealpha=0.94,
    )
    mean_score = float(score.mean(skipna=True))
    upper.set_title(
        f"Level: mean daily score {mean_score:.1f} over the sample "
        f"({int(score.notna().sum())} valid trading days)",
        fontsize=10,
    )

    lower = axes[1]
    # NaN bars are simply not drawn, so a warm-up day or a no-news day stays
    # blank instead of being flattened to a zero-height bar at the axis.
    colours = np.where(
        anomaly > 0, SENTIMENT_POSITIVE_COLOR,
        np.where(anomaly < 0, SENTIMENT_NEGATIVE_COLOR, "#8A8F98"),
    )
    lower.bar(dates, anomaly, color=colours, width=1.0, linewidth=0)
    lower.axhline(0, color="#4B5563", linewidth=0.9)
    lower.set_ylabel("Mean sector expanding z-score")
    lower.set_xlabel("Date")
    lower.tick_params(axis="x", labelrotation=30)
    lower.set_title(
        "Anomaly: equal-weight mean of the sectors' own expanding z-scores "
        "(not re-standardised)",
        fontsize=10,
    )

    start_date = dates.min()
    end_date = dates.max()
    upper.set_xlim(start_date, end_date)
    lower.set_xlim(start_date, end_date)

    add_figure_heading(
        "Figure 9. Market-wide Extended FinVADER sentiment, 2021-2023",
        "Equal-weight sector aggregate: sentiment level and relative historical position",
        fig=fig,
    )
    path = FIGURES / "figure_9_market_wide_sentiment_index.png"
    return finalise_figure(
        path, EXTENDED_SENTIMENT_SOURCE, _date_range(aggregate, "date"), top=0.95
    )


def sector_sentiment_ranking_table(sector_index: pd.DataFrame) -> pd.DataFrame:
    """Descriptive mean/median Extended compound per sector, best first.

    Missing sector-days are excluded rather than counted as neutral, and
    sectors are not weighted by how much news they attracted - a heavily
    covered sector and a thin one each contribute one average.
    """
    valid = sector_index.dropna(subset=["extended_compound"])
    ranking = (
        valid.groupby("sector")["extended_compound"]
        .agg(mean_extended_compound="mean", median_extended_compound="median",
             n_valid_sector_days="size")
        .reset_index()
        .sort_values("mean_extended_compound", ascending=False)
        .reset_index(drop=True)
    )
    return ranking


def plot_sector_sentiment_ranking(sector_index: pd.DataFrame) -> pathlib.Path:
    """Figure 10: average Extended FinVADER sentiment by equity sector."""
    sentiment.assert_application_period_only(
        sector_index["date"], context="figure_10 sector_index input"
    )
    ranking = sector_sentiment_ranking_table(sector_index)

    # Highest mean at the top: barh draws upwards, so plot the reversed order.
    ordered = ranking.iloc[::-1].reset_index(drop=True)
    positions = np.arange(len(ordered))

    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    ax.hlines(
        positions, 0.0, ordered["mean_extended_compound"],
        color=SENTIMENT_ROLLING_COLOR, linewidth=2.2, alpha=0.85,
    )
    ax.scatter(
        ordered["mean_extended_compound"], positions, s=95,
        color=SENTIMENT_ROLLING_COLOR, zorder=3, label="Mean",
    )
    ax.scatter(
        ordered["median_extended_compound"], positions, s=70, facecolors="none",
        edgecolors="#6B625C", linewidths=1.2, zorder=4, label="Median",
    )
    ax.axvline(0.0, color="#4B5563", linewidth=1.0, linestyle="--")

    ax.set_yticks(positions)
    ax.set_yticklabels(ordered["sector"])
    ax.set_xlabel("Extended FinVADER compound score")
    ax.set_ylabel("Equity sector")

    span = float(ranking["mean_extended_compound"].max())
    for position, value in zip(positions, ordered["mean_extended_compound"]):
        ax.text(
            value + span * 0.03, position, f"{value:.3f}",
            va="center", ha="left", fontsize=8.5, color="#1F2933",
        )
    # Headroom so the value labels never collide with the right-hand spine.
    ax.set_xlim(0.0, span * 1.22)
    ax.legend(
        loc="lower right", fontsize=8.5, frameon=True, facecolor="white",
        edgecolor="#D8DDE6", framealpha=0.94,
    )

    day_counts = ranking["n_valid_sector_days"]
    add_figure_heading(
        "Figure 10. Average Extended FinVADER sentiment by equity sector, 2021-2023",
        "Descriptive mean and median of the raw compound score over each sector's valid sector-days "
        f"({int(day_counts.min())}-{int(day_counts.max())} days per sector);\n"
        "missing sector-days are excluded rather than treated as neutral, and sectors are not weighted "
        "by headline count. Differences between sectors are\n"
        "descriptive only - they are not a significance test, a causal claim, or a forecast of future returns.",
        fig=fig,
    )
    path = FIGURES / "figure_10_sector_sentiment_ranking.png"
    return finalise_figure(
        path, EXTENDED_SENTIMENT_SOURCE, _date_range(sector_index, "date"), top=0.90
    )


def run_sentiment_build() -> dict[str, pd.DataFrame]:
    """Build the 2021-2023 extended sentiment index and comparison tables.

    Independent of the fund pipeline above (only shares data_access/etl/
    features and the results/ paths) and independent of the unimplemented
    src/fusion.py, so it can run - and be tested - on its own.
    """
    ensure_dirs()
    apply_theme()

    print("Loading headlines and equities for the sentiment build ...")
    equities_raw = data_access.load_equity_prices()
    headlines_raw = data_access.load_news_headlines()
    clean_equities, _ = etl.clean_price_panel(equities_raw, "equity_prices", has_sector=True)
    clean_headlines, _ = etl.clean_headlines(headlines_raw)
    equity_calendar = clean_equities["date"].drop_duplicates().sort_values()

    headline_panel, _ = features.assemble_headline_panel(clean_headlines, equity_calendar)
    headline_panel = headline_panel.copy()
    headline_panel["raw_title"] = headline_panel["text_raw"]

    app_start = pd.Timestamp(sentiment.APPLICATION_START_DATE)
    app_end = pd.Timestamp(sentiment.APPLICATION_END_DATE)
    in_application_period = headline_panel["trading_date"].between(app_start, app_end, inclusive="both")
    application_headlines = headline_panel.loc[in_application_period].reset_index(drop=True)
    print(
        f"Application-period headlines (2021-01-01 to 2023-12-31, by trading_date): "
        f"{len(application_headlines):,} of {len(headline_panel):,} aligned headlines"
    )

    print("Scoring distinct raw titles with baseline and extended FinVADER ...")
    scored = sentiment.score_headlines_dual(application_headlines)
    sentiment.assert_application_period_only(
        scored["trading_date"], context="scored 2021-2023 headlines"
    )

    ticker_day = sentiment.ticker_day_sentiment(scored)
    sector_universe = clean_equities.groupby("sector")["ticker"].nunique().to_dict()
    trading_dates_2021_2023 = equity_calendar.loc[
        equity_calendar.between(app_start, app_end, inclusive="both")
    ]
    sector_day = sentiment.sector_day_sentiment(
        ticker_day, trading_dates=trading_dates_2021_2023, sector_universe=sector_universe
    )
    sector_index = sentiment.add_expanding_zscores(
        sector_day, min_periods=SECTOR_INDEX_MIN_PERIODS
    )
    sentiment.assert_application_period_only(sector_index["date"], context="sector_sentiment_index")

    save_csv(sector_index, DATA / "sector_sentiment_index.csv")

    print("Building coverage, impact, and comparison tables ...")
    coverage = sentiment.custom_lexicon_coverage_table(scored)
    impact = sentiment.custom_term_impact_table(scored)
    headline_cmp = sentiment.headline_sentiment_comparison_table(scored)
    largest_changes = sentiment.largest_sentiment_score_changes_table(scored)
    neutral_summary = sentiment.neutral_reclassification_summary_table(scored)
    sector_cmp = sentiment.sector_sentiment_model_comparison_table(sector_index, scored)

    save_csv(coverage, TABLES / "custom_lexicon_coverage.csv")
    save_csv(impact, TABLES / "custom_term_impact.csv")
    save_csv(headline_cmp, TABLES / "headline_sentiment_comparison.csv")
    save_csv(largest_changes, TABLES / "largest_sentiment_score_changes.csv")
    save_csv(neutral_summary, TABLES / "neutral_reclassification_summary.csv")
    save_csv(sector_cmp, TABLES / "sector_sentiment_model_comparison.csv")

    figure_paths = [
        plot_extended_sector_sentiment_index(sector_index),
        plot_baseline_vs_extended_sentiment(sector_index, sector_cmp),
        # Report-only exhibits, drawn from the same in-memory sector index.
        plot_market_wide_sentiment_index(sector_index),
        plot_sector_sentiment_ranking(sector_index),
    ]
    for path in figure_paths:
        print(f"wrote {path.relative_to(ROOT)}")

    print("\nSentiment build complete (2021-2023 application period).")
    return {
        "scored_headlines": scored,
        "ticker_day": ticker_day,
        "sector_index": sector_index,
        "coverage": coverage,
        "impact": impact,
        "headline_comparison": headline_cmp,
        "largest_changes": largest_changes,
        "neutral_summary": neutral_summary,
        "sector_comparison": sector_cmp,
    }


# ============================================================================
# Station 3 - Fusion: extended sentiment tilt on Equity-Only Minimum Variance
#
# Only the frozen extended lexicon is used here; the baseline-vs-extended
# comparison belongs to the sentiment module and is not repeated at the
# portfolio level. Tilt strength is the fixed symmetric pair lambda = +1
# (momentum) and lambda = -1 (contrarian), with lambda = 0 being the existing
# untilted fund reused as-is. All three are priced by the same shared backtest
# executor with the same transaction-cost assumption.
# ============================================================================

BASE_FUND_FAMILY = "Equity-Only"
BASE_FUND_METHOD = "minimum_variance"
BASE_FUND_NAME = "Equity-Only Minimum Variance"
BASE_VARIANT_NAME = "Equity-Only Minimum Variance"
MOMENTUM_VARIANT_NAME = "Extended Sentiment Momentum"
CONTRARIAN_VARIANT_NAME = "Extended Sentiment Contrarian"
EQUITY_ANNUALIZATION_FACTOR = 252
FUSION_TRANSACTION_COST_RATE = 0.001

# (lambda, direction, variant label) for the two tilted variants.
FUSION_TILT_VARIANTS = [
    (1.0, "momentum", MOMENTUM_VARIANT_NAME),
    (-1.0, "contrarian", CONTRARIAN_VARIANT_NAME),
]
FUSION_VARIANT_ORDER = [
    BASE_VARIANT_NAME, MOMENTUM_VARIANT_NAME, CONTRARIAN_VARIANT_NAME
]

FUSION_SOURCE = (
    "FINS3645 equity price and news-headline bundles; RiskBridge extended "
    "sentiment tilt on the Equity-Only Minimum Variance fund, priced with the "
    "shared walk-forward backtest engine (Part B, Station 3, 2021-2023)"
)

FUSION_VARIANT_STYLE = {
    # Neutral grey for the untilted benchmark; the two tilt directions share a
    # family of warm/cool tones so they read as a matched pair.
    BASE_VARIANT_NAME: {"color": "#4A5568", "linestyle": "-", "linewidth": 2.2},
    MOMENTUM_VARIANT_NAME: {"color": "#B23A48", "linestyle": "-", "linewidth": 1.5},
    CONTRARIAN_VARIANT_NAME: {"color": "#2E86C1", "linestyle": "--", "linewidth": 1.5},
}


def _base_weights_by_effective_date(
    fund_weights: pd.DataFrame,
) -> tuple[dict[pd.Timestamp, pd.Series], dict[pd.Timestamp, pd.Timestamp]]:
    """Read the base fund's existing rebalance schedule out of fund_weights.

    The monthly rebalance dates come from the fund that already exists rather
    than being re-derived, so the tilt cannot end up trading on a different
    calendar from the fund it is compared against.
    """
    base = fund_weights.loc[
        (fund_weights["fund_family"] == BASE_FUND_FAMILY)
        & (fund_weights["method"] == BASE_FUND_METHOD)
    ].copy()
    if base.empty:
        raise ValueError(f"{BASE_FUND_NAME} not found in fund_weights")
    base["formation_date"] = pd.to_datetime(base["formation_date"])
    base["effective_date"] = pd.to_datetime(base["effective_date"])

    weights_by_effective: dict[pd.Timestamp, pd.Series] = {}
    formation_by_effective: dict[pd.Timestamp, pd.Timestamp] = {}
    for effective_date, group in base.groupby("effective_date"):
        if group["ticker"].duplicated().any():
            raise ValueError(f"duplicate tickers in base weights on {effective_date}")
        series = pd.Series(
            group["target_weight"].to_numpy(dtype=float),
            index=pd.Index(group["ticker"].astype(str), name="ticker"),
        ).sort_index()
        weights_by_effective[pd.Timestamp(effective_date)] = series
        formation_by_effective[pd.Timestamp(effective_date)] = pd.Timestamp(
            group["formation_date"].iloc[0]
        )
    return weights_by_effective, formation_by_effective


def build_extended_coverage(
    ticker_z: pd.DataFrame,
    base_weights_by_effective: Mapping[pd.Timestamp, pd.Series],
    formation_by_effective: Mapping[pd.Timestamp, pd.Timestamp],
    signal_by_effective: Mapping[pd.Timestamp, pd.Timestamp],
) -> pd.DataFrame:
    """Per-rebalance extended-signal coverage, one row per formation date.

    Coverage is reported as found. A thin month stays thin: relaxing the
    exact-match signal rule to raise it would mean trading on a stale reading
    presented as current information.
    """
    rows: list[dict[str, object]] = []
    for effective_date in sorted(base_weights_by_effective):
        base = base_weights_by_effective[effective_date]
        signal_date = pd.Timestamp(signal_by_effective[effective_date])
        positive = base > portfolios.WEIGHT_TOLERANCE
        z = fusion.lookup_ticker_signal(ticker_z, signal_date, base.index)
        has_z = z.notna()
        n_positive = int(positive.sum())
        n_positive_with_z = int((positive & has_z).sum())
        rows.append(
            {
                "formation_date": pd.Timestamp(formation_by_effective[effective_date]),
                "effective_date": pd.Timestamp(effective_date),
                "signal_date": signal_date,
                "lexicon": "extended",
                "n_tickers_total": len(base),
                "n_tickers_with_z": int(has_z.sum()),
                "coverage_ratio": float(has_z.mean()) if len(base) else np.nan,
                "n_positive_base_weight": n_positive,
                "n_positive_base_weight_with_z": n_positive_with_z,
                "positive_weight_coverage_ratio": (
                    n_positive_with_z / n_positive if n_positive else np.nan
                ),
                "base_weight_with_z": float(base.loc[has_z].sum()),
                # tilt_active records whether a usable signal reached at least
                # one held position; it is independent of the tilt direction.
                "tilt_active": bool(n_positive_with_z > 0),
            }
        )
    return pd.DataFrame(rows)


def _turnover_summary(turnover_by_effective: pd.Series) -> dict[str, float]:
    """Split turnover into the initial build and the ongoing rebalances.

    The first rebalance always has turnover 1.0 because the book is built from
    cash; averaging it in with ongoing rebalances would overstate steady-state
    trading.
    """
    ordered = turnover_by_effective.sort_index().dropna()
    if ordered.empty:
        return {
            "initial_turnover": np.nan,
            "avg_rebalance_turnover_ex_initial": np.nan,
            "total_turnover": np.nan,
        }
    subsequent = ordered.iloc[1:]
    return {
        "initial_turnover": float(ordered.iloc[0]),
        "avg_rebalance_turnover_ex_initial": (
            float(subsequent.mean()) if len(subsequent) else np.nan
        ),
        "total_turnover": float(ordered.sum()),
    }


def _cumulative_returns(returns: pd.DataFrame) -> dict[str, float]:
    """Compound daily returns; never sum them.

    growth_of_1 is built from net returns only, so its final value and the
    cumulative net return are two views of one number and are checked against
    each other. The cumulative GROSS return is compounded from gross returns
    independently, never derived from the net growth series.
    """
    gross = pd.to_numeric(returns["gross_return"], errors="raise")
    net = pd.to_numeric(returns["net_return"], errors="raise")
    cumulative_gross = float((1.0 + gross).prod() - 1.0)
    cumulative_net = float((1.0 + net).prod() - 1.0)
    growth = float((1.0 + net).cumprod().iloc[-1])
    if not np.isclose(growth, 1.0 + cumulative_net, rtol=1e-12, atol=1e-12):
        raise AssertionError(
            f"growth_of_1 final value {growth!r} disagrees with "
            f"1 + cumulative_net_return {1.0 + cumulative_net!r}"
        )
    return {
        "cumulative_gross_return": cumulative_gross,
        "cumulative_net_return": cumulative_net,
    }


def _variant_weight_diagnostics(
    detail: pd.DataFrame, base_weights_by_effective: Mapping[pd.Timestamp, pd.Series]
) -> dict[str, float]:
    """Concentration and distance-from-base diagnostics for one tilt variant.

    effective_number_of_holdings and l1_distance_from_base are averaged over
    rebalances; max_stock_weight and max_absolute_weight_change are maxima over
    every (rebalance, ticker) pair.
    """
    per_rebalance_enh: list[float] = []
    per_rebalance_l1: list[float] = []
    max_abs_change = 0.0
    for effective_date, group in detail.groupby("effective_date"):
        weights = group["target_weight"].to_numpy(dtype=float)
        sum_sq = float((weights**2).sum())
        per_rebalance_enh.append(1.0 / sum_sq if sum_sq > 0 else np.nan)
        base = base_weights_by_effective[pd.Timestamp(effective_date)]
        aligned = group.set_index("ticker")["target_weight"].reindex(base.index)
        diff = (aligned - base).abs()
        per_rebalance_l1.append(float(diff.sum()))
        max_abs_change = max(max_abs_change, float(diff.max()))

    n_weights = len(detail)
    n_clipped = int(detail["clipped_to_zero"].sum())
    n_fallback_rebalances = int(
        detail.groupby("effective_date")["fallback_used"].any().sum()
    )
    return {
        "n_weights_clipped_to_zero": n_clipped,
        "share_weights_clipped_to_zero": n_clipped / n_weights if n_weights else np.nan,
        "n_fallback_rebalances": n_fallback_rebalances,
        "max_stock_weight": float(detail["target_weight"].max()),
        "effective_number_of_holdings": float(np.nanmean(per_rebalance_enh)),
        "l1_distance_from_base": float(np.mean(per_rebalance_l1)),
        "max_absolute_weight_change": max_abs_change,
    }


def plot_fusion_before_vs_after(fusion_returns: pd.DataFrame) -> pathlib.Path:
    """Figure 8: growth of $1 for the base fund and two extended tilts.

    Word-caption detail: all three series use the same Equity-Only Minimum
    Variance book, backtest engine, rebalance dates, and transaction-cost
    assumption. They differ only in the sentiment tilt applied to target
    weights. The +1 and -1 directions were fixed in advance rather than
    selected after observing these results.
    """
    sentiment.assert_application_period_only(
        fusion_returns["date"], context="figure_8 fusion_returns input"
    )
    plt.figure(figsize=(10.0, 5.8))
    for variant in FUSION_VARIANT_ORDER:
        series = fusion_returns.loc[fusion_returns["variant"] == variant].sort_values("date")
        if series.empty:
            continue
        plt.plot(
            series["date"], series["growth_of_1"], label=variant,
            **FUSION_VARIANT_STYLE[variant],
        )
    plt.axhline(1.0, color="#4B5563", linewidth=0.6, linestyle=":")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1 (net of transaction costs)")
    add_figure_heading(
        "Figure 8. Extended sentiment tilt versus the untilted base fund, 2021-2023",
        "Equity-Only Minimum Variance, net of transaction costs",
    )
    plt.legend(loc="upper left", fontsize=8, frameon=True, facecolor="white",
               edgecolor="#D8DDE6", framealpha=0.94)
    path = FIGURES / "figure_8_fusion_before_vs_after.png"
    return finalise_figure(
        path, FUSION_SOURCE, _date_range(fusion_returns, "date"), top=0.94
    )


def run_fusion_build(
    ticker_day: pd.DataFrame,
    equity_returns_wide: pd.DataFrame,
    equity_universe: pd.DataFrame,
    equity_trading_dates: pd.Series,
) -> dict[str, pd.DataFrame]:
    """Apply the extended sentiment tilt to the base fund and save the outputs.

    Consumes the ticker-day sentiment already produced by the sentiment build;
    it never reloads headlines or re-scores them. The lambda = 0 case is the
    existing base fund read back from disk, not a re-run of an equivalent
    backtest.
    """
    ensure_dirs()
    apply_theme()
    print("\nBuilding extended sentiment-tilt fusion variants ...")

    full_calendar = pd.DatetimeIndex(
        sorted(pd.to_datetime(pd.Series(list(equity_trading_dates))).dt.normalize().unique())
    )
    app_start = pd.Timestamp(sentiment.APPLICATION_START_DATE)
    app_end = pd.Timestamp(sentiment.APPLICATION_END_DATE)
    application_dates = full_calendar[
        (full_calendar >= app_start) & (full_calendar <= app_end)
    ]
    universe_tickers = sorted(equity_universe["ticker"].astype(str).unique())
    sector_by_ticker = (
        equity_universe.drop_duplicates("ticker")
        .set_index("ticker")["sector"]
        .astype(str)
        .to_dict()
    )

    ticker_z = fusion.build_ticker_extended_z(
        ticker_day, application_dates, universe_tickers
    )
    save_csv(ticker_z, DATA / "ticker_sentiment_z.csv")
    missing_share = float(ticker_z["extended_z_expanding"].isna().mean())
    print(
        f"ticker extended z grid: {len(ticker_z):,} rows "
        f"({len(application_dates)} trading days x {len(universe_tickers)} tickers), "
        f"{missing_share:.1%} missing"
    )

    fund_weights = pd.read_csv(DATA / "fund_weights.csv")
    fund_returns = pd.read_csv(DATA / "fund_returns.csv", parse_dates=["date"])
    fund_metrics = pd.read_csv(TABLES / "performance_metrics.csv")
    fund_diagnostics = pd.read_csv(
        TABLES / "fund_backtest_diagnostics.csv",
        parse_dates=["formation_date", "effective_date"],
    )

    base_weights_by_effective, formation_by_effective = _base_weights_by_effective_date(
        fund_weights
    )
    signal_by_effective = {
        effective_date: fusion.previous_trading_day(effective_date, full_calendar)
        for effective_date in base_weights_by_effective
    }

    coverage = build_extended_coverage(
        ticker_z, base_weights_by_effective, formation_by_effective, signal_by_effective
    )
    save_csv(coverage, TABLES / "ticker_sentiment_coverage_by_formation_date.csv")

    # ---- base variant: reuse the existing fund, never re-run it -----------
    base_returns = fund_returns.loc[fund_returns["fund_name"] == BASE_FUND_NAME].copy()
    base_metrics_row = fund_metrics.loc[fund_metrics["fund_name"] == BASE_FUND_NAME].iloc[0]
    base_diag = fund_diagnostics.loc[fund_diagnostics["fund_name"] == BASE_FUND_NAME]
    base_turnover = _turnover_summary(base_diag.set_index("effective_date")["turnover"])

    weight_frames: list[pd.DataFrame] = []
    return_frames: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    base_weight_frames = []
    for effective_date, base in base_weights_by_effective.items():
        base_weight_frames.append(
            pd.DataFrame(
                {
                    "formation_date": pd.Timestamp(formation_by_effective[effective_date]),
                    "effective_date": pd.Timestamp(effective_date),
                    "signal_date": pd.Timestamp(signal_by_effective[effective_date]),
                    "variant": BASE_VARIANT_NAME,
                    "lexicon": "none",
                    "lambda": 0.0,
                    "direction": "base",
                    "ticker": base.index.astype(str),
                    "base_weight": base.to_numpy(dtype=float),
                    "ticker_z": np.nan,
                    "tilt_multiplier": 1.0,
                    "pre_normalisation_weight": base.to_numpy(dtype=float),
                    "target_weight": base.to_numpy(dtype=float),
                    "signal_available": False,
                    "clipped_to_zero": False,
                    "fallback_used": False,
                }
            )
        )
    base_weights_frame = pd.concat(base_weight_frames, ignore_index=True)
    base_weights_frame["sector"] = base_weights_frame["ticker"].map(sector_by_ticker)
    weight_frames.append(base_weights_frame)

    base_returns_out = base_returns[
        ["date", "gross_return", "transaction_cost", "net_return", "growth_of_1", "drawdown"]
    ].copy()
    base_returns_out.insert(1, "variant", BASE_VARIANT_NAME)
    base_returns_out.insert(2, "lexicon", "none")
    base_returns_out.insert(3, "lambda", 0.0)
    base_returns_out.insert(4, "direction", "base")
    return_frames.append(base_returns_out)

    base_cumulative = _cumulative_returns(base_returns_out)
    summary_rows.append(
        {
            "variant": BASE_VARIANT_NAME,
            "lexicon": "none",
            "lambda": 0.0,
            "direction": "base",
            "annual_return": float(base_metrics_row["annual_return"]),
            "annual_volatility": float(base_metrics_row["annual_volatility"]),
            "sharpe_ratio": float(base_metrics_row["sharpe_ratio"]),
            "max_drawdown": float(base_metrics_row["max_drawdown"]),
            **base_cumulative,
            **base_turnover,
        }
    )
    diagnostic_rows.append(
        {
            "variant": BASE_VARIANT_NAME,
            "lexicon": "none",
            "lambda": 0.0,
            "direction": "base",
            **base_turnover,
            # The base fund is untilted, so the tilt-specific diagnostics are
            # structurally zero rather than unknown.
            "n_weights_clipped_to_zero": 0,
            "share_weights_clipped_to_zero": 0.0,
            "n_fallback_rebalances": 0,
            "max_stock_weight": float(base_weights_frame["target_weight"].max()),
            "effective_number_of_holdings": float(
                np.mean(
                    [
                        1.0 / float((w.to_numpy(dtype=float) ** 2).sum())
                        for w in base_weights_by_effective.values()
                    ]
                )
            ),
            "l1_distance_from_base": 0.0,
            "max_absolute_weight_change": 0.0,
        }
    )

    # ---- the two extended tilt variants ----------------------------------
    for lam, direction, variant in FUSION_TILT_VARIANTS:
        targets, detail = fusion.build_tilted_target_schedule(
            base_weights_by_effective, signal_by_effective, ticker_z, lam
        )
        executed = portfolios.run_backtest_from_target_schedule(
            equity_returns_wide,
            targets,
            formation_by_effective,
            transaction_cost_rate=FUSION_TRANSACTION_COST_RATE,
        )

        detail = detail.copy()
        detail["formation_date"] = detail["effective_date"].map(formation_by_effective)
        detail["variant"] = variant
        detail["lexicon"] = "extended"
        detail["lambda"] = lam
        detail["direction"] = direction
        detail["sector"] = detail["ticker"].map(sector_by_ticker)
        weight_frames.append(detail)

        returns_out = executed.returns.copy()
        growth = portfolios.calculate_growth_of_1(returns_out["net_return"])
        returns_out["growth_of_1"] = growth.reindex(returns_out.index)
        returns_out["drawdown"] = portfolios._drawdown_from_growth(returns_out["growth_of_1"])
        returns_out = returns_out.reset_index()
        returns_out.insert(1, "variant", variant)
        returns_out.insert(2, "lexicon", "extended")
        returns_out.insert(3, "lambda", lam)
        returns_out.insert(4, "direction", direction)
        return_frames.append(returns_out)

        metrics = portfolios.calculate_performance_metrics(
            executed.returns["net_return"], EQUITY_ANNUALIZATION_FACTOR
        )
        turnover = _turnover_summary(executed.diagnostics.set_index("effective_date")["turnover"])
        cumulative = _cumulative_returns(returns_out)
        summary_rows.append(
            {
                "variant": variant, "lexicon": "extended", "lambda": lam,
                "direction": direction, **metrics, **cumulative, **turnover,
            }
        )
        diagnostic_rows.append(
            {
                "variant": variant, "lexicon": "extended", "lambda": lam,
                "direction": direction, **turnover,
                **_variant_weight_diagnostics(detail, base_weights_by_effective),
            }
        )

    fusion_weights = pd.concat(weight_frames, ignore_index=True)[
        [
            "formation_date", "effective_date", "signal_date", "variant", "lexicon",
            "lambda", "direction", "ticker", "sector", "base_weight", "ticker_z",
            "tilt_multiplier", "pre_normalisation_weight", "target_weight",
            "signal_available", "clipped_to_zero", "fallback_used",
        ]
    ]
    fusion_returns = pd.concat(return_frames, ignore_index=True)[
        [
            "date", "variant", "lexicon", "lambda", "direction", "gross_return",
            "transaction_cost", "net_return", "growth_of_1", "drawdown",
        ]
    ]
    fusion_diagnostics = pd.DataFrame(diagnostic_rows)
    fusion_summary = pd.DataFrame(summary_rows)[
        [
            "variant", "lexicon", "lambda", "direction", "annual_return",
            "annual_volatility", "sharpe_ratio", "max_drawdown",
            "cumulative_gross_return", "cumulative_net_return", "initial_turnover",
            "avg_rebalance_turnover_ex_initial", "total_turnover",
        ]
    ]

    sentiment.assert_application_period_only(fusion_returns["date"], context="fusion_returns")
    if len(fusion_summary) != 3:
        raise AssertionError(
            f"fusion summary must have exactly 3 rows, got {len(fusion_summary)}"
        )

    save_csv(fusion_weights, DATA / "fusion_weights.csv")
    save_csv(fusion_returns, DATA / "fusion_returns.csv")
    save_csv(fusion_diagnostics, TABLES / "fusion_diagnostics.csv")
    save_csv(fusion_summary, TABLES / "fusion_before_vs_after.csv")
    print(f"wrote {plot_fusion_before_vs_after(fusion_returns).relative_to(ROOT)}")

    print("\nFusion build complete (extended lexicon, lambda in {-1, 0, +1}).")
    return {
        "ticker_z": ticker_z,
        "coverage": coverage,
        "fusion_weights": fusion_weights,
        "fusion_returns": fusion_returns,
        "fusion_diagnostics": fusion_diagnostics,
        "fusion_summary": fusion_summary,
    }


# ============================================================================
# Station 3 - Sentiment: FinVADER 2020 candidate-term discovery
# ============================================================================

CANDIDATE_PATH = TABLES / "sentiment_candidate_terms.csv"


def _discovery_rows(panel: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(panel["date"], errors="coerce")
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_convert(None)
    dates = dates.dt.normalize()
    mask = dates.between("2020-01-01", "2020-12-31", inclusive="both")
    return panel.loc[mask].copy()


def run_sentiment_candidate_discovery() -> pd.DataFrame:
    """Build and save candidate terms using only 2020 raw headlines."""
    equities = data_access.load_equity_prices()
    headlines = data_access.load_news_headlines()
    clean_equities, _ = etl.clean_price_panel(
        equities, "equity_prices", has_sector=True
    )
    clean_headlines, _ = etl.clean_headlines(headlines)
    equity_calendar = clean_equities["date"].drop_duplicates().sort_values()
    headline_panel, _ = features.assemble_headline_panel(
        clean_headlines, equity_calendar
    )

    headline_panel = headline_panel.copy()
    headline_panel["raw_title"] = headline_panel["text_raw"]
    retained_columns = ["raw_title", "date", "ticker", "sector", "trading_date"]
    headline_panel = headline_panel[retained_columns]

    discovery = _discovery_rows(headline_panel)
    source_dates = pd.to_datetime(discovery["date"], errors="coerce")
    if getattr(source_dates.dt, "tz", None) is not None:
        source_dates = source_dates.dt.tz_convert(None)
    source_dates = source_dates.dt.normalize()
    outside_window = int(
        (~source_dates.between("2020-01-01", "2020-12-31", inclusive="both")).sum()
    )
    if outside_window != 0:
        raise AssertionError(
            f"candidate source contains {outside_window} row(s) outside 2020"
        )

    candidates = sentiment.candidate_terms(
        discovery,
        text_col="raw_title",
        date_col="date",
        start_date="2020-01-01",
        end_date="2020-12-31",
        top_n=None,
    )
    TABLES.mkdir(parents=True, exist_ok=True)
    restored = sentiment.write_candidate_terms(candidates, CANDIDATE_PATH)

    metadata = sentiment.finvader_metadata()
    uncovered = candidates.loc[
        ~candidates["in_lexicon"] & candidates["review_eligible"]
    ].head(30)

    try:
        display_path = CANDIDATE_PATH.relative_to(ROOT)
    except ValueError:
        display_path = CANDIDATE_PATH
    print(f"wrote {display_path}")
    print(f"candidate rows: {len(candidates)}")
    print(f"round-trip rows: {len(restored)}")
    print(f"round-trip literal 'nan' terms: {int(restored['term'].eq('nan').sum())}")
    print(f"round-trip missing terms: {int(restored['term'].isna().sum())}")
    print(
        "FinVADER baseline: "
        f"version={metadata['finvader_version']}, "
        f"indicator={metadata['indicator']}, "
        f"use_sentibignomics={metadata['use_sentibignomics']}, "
        f"use_henry={metadata['use_henry']}"
    )
    if discovery.empty:
        print("discovery headlines: 0")
        print("discovery date range: none")
        print("discovery dates: 0, tickers: 0, sectors: 0")
    else:
        discovery_dates = source_dates
        print(f"discovery headlines: {len(discovery)}")
        print(
            "discovery date range: "
            f"{discovery_dates.min().date()} to {discovery_dates.max().date()}"
        )
        print(
            f"discovery dates: {discovery_dates.nunique()}, "
            f"tickers: {discovery['ticker'].nunique()}, "
            f"sectors: {discovery['sector'].nunique()}"
        )
    print(f"candidate source rows outside discovery window: {outside_window}")
    print("top 30 eligible uncovered candidate terms:")
    display_columns = [
        "term",
        "frequency",
        "document_frequency",
        "n_tickers",
        "n_sectors",
    ]
    print(uncovered[display_columns].to_string(index=False))
    return candidates


# ============================================================================
# CLI entry point
# ============================================================================


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sentiment-candidates-only",
        action="store_true",
        help="build the frozen 2020 sentiment candidate-term table and exit",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.sentiment_candidates_only:
        run_sentiment_candidate_discovery()
        return
    run_full_build()


if __name__ == "__main__":
    main()
