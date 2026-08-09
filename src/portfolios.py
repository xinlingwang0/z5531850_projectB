"""Portfolio construction and walk-forward backtesting for Project B.

This module is intentionally pure algorithm code: it does not read files, write
files, download data, or generate figures. It consumes date-by-ticker return
panels produced by ``src.features`` and returns DataFrames that orchestration
code can save under ``results/``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform


METHODS: tuple[str, ...] = (
    "equal_weight",
    "minimum_variance",
    "maximum_sharpe",
    "hrp",
)
METHOD_LABELS: dict[str, str] = {
    "equal_weight": "Equal Weight",
    "minimum_variance": "Minimum Variance",
    "maximum_sharpe": "Maximum Sharpe",
    "hrp": "HRP",
}
FAMILY_CONFIG: dict[str, dict[str, int | str]] = {
    "Equity-Only": {"window": 252, "annualization_factor": 252},
    "Crypto-Only": {"window": 365, "annualization_factor": 365},
    "Combined": {"window": 252, "annualization_factor": 252},
}
WEIGHT_TOLERANCE = 1e-6
RISK_EPSILON = 1e-12


@dataclass
class BacktestResult:
    """Container returned by ``run_walk_forward_backtest``.

    Attributes
    ----------
    returns:
        Daily gross, cost, and net fund returns indexed by date.
    weights:
        Target weights by formation date, effective date, and ticker.
    diagnostics:
        Per-rebalance solver, fallback, turnover, and transaction-cost records.
    """

    returns: pd.DataFrame
    weights: pd.DataFrame
    diagnostics: pd.DataFrame


class PortfolioConstructionError(ValueError):
    """Raised when a portfolio method cannot produce valid weights."""


def _normalise_method(method: str) -> str:
    aliases = {
        "equal": "equal_weight",
        "equal_weight": "equal_weight",
        "ew": "equal_weight",
        "min_variance": "minimum_variance",
        "minimum_variance": "minimum_variance",
        "min_var": "minimum_variance",
        "maximum_sharpe": "maximum_sharpe",
        "max_sharpe": "maximum_sharpe",
        "tangency": "maximum_sharpe",
        "mean_variance_tangency": "maximum_sharpe",
        "hrp": "hrp",
        "hierarchical_risk_parity": "hrp",
    }
    key = method.strip().lower().replace("-", "_").replace(" ", "_")
    if key not in aliases:
        raise ValueError(f"unknown portfolio method '{method}'")
    return aliases[key]


def _validate_positive_integer(value: object, name: str) -> int:
    if not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _empty_returns_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["gross_return", "transaction_cost", "net_return"])


def _empty_weights_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["formation_date", "effective_date", "ticker", "target_weight"]
    )


def _empty_diagnostics_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "formation_date",
            "effective_date",
            "method",
            "n_assets",
            "solver_success",
            "fallback_used",
            "message",
            "turnover",
            "transaction_cost",
        ]
    )


def _validate_returns_frame(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Validate and copy a date-by-ticker return panel.

    NaNs are allowed at this stage because ``src.features`` can produce aligned
    panels with sparse starts. Rebalance eligibility later requires finite values
    over the full estimation window.
    """
    if not isinstance(returns_df, pd.DataFrame):
        raise TypeError("returns_df must be a pandas DataFrame")
    if returns_df.empty:
        raise ValueError("returns_df must not be empty")
    if not isinstance(returns_df.index, pd.DatetimeIndex):
        raise ValueError("returns_df must use a DatetimeIndex")
    if returns_df.index.has_duplicates:
        raise ValueError("returns_df index must contain unique dates")
    if not returns_df.index.is_monotonic_increasing:
        raise ValueError("returns_df index must be strictly increasing")
    if returns_df.columns.has_duplicates:
        raise ValueError("returns_df columns must be unique ticker names")

    out = returns_df.copy()
    out.columns = pd.Index(str(col).strip() for col in out.columns)
    if any(col == "" for col in out.columns):
        raise ValueError("returns_df contains blank ticker names")
    if out.columns.has_duplicates:
        raise ValueError("returns_df contains duplicate ticker names after normalisation")
    try:
        out = out.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("returns_df must contain numeric values only") from exc
    if np.isinf(out.to_numpy(dtype=float)).any():
        raise ValueError("returns_df must not contain inf or -inf values")
    if (out <= -1.0).any().any():
        raise ValueError("returns_df contains returns <= -100%, which cannot be backtested safely")
    return out


def _prepare_weight_input(returns_df: pd.DataFrame) -> pd.DataFrame:
    clean = _validate_returns_frame(returns_df)
    if clean.isna().any().any():
        raise PortfolioConstructionError(
            "portfolio weight methods require a complete finite estimation sample"
        )
    if clean.shape[1] == 0:
        raise PortfolioConstructionError("at least one asset is required")
    if clean.shape[0] < 2:
        raise PortfolioConstructionError("at least two observations are required")
    return clean


def _normalise_weight_array(weights: np.ndarray, columns: pd.Index) -> pd.Series:
    arr = np.asarray(weights, dtype=float).copy()
    if arr.ndim != 1 or arr.shape[0] != len(columns):
        raise PortfolioConstructionError("weight vector has the wrong shape")
    if not np.isfinite(arr).all():
        raise PortfolioConstructionError("weight vector contains NaN or inf")
    arr[np.abs(arr) < WEIGHT_TOLERANCE] = 0.0
    if (arr < -WEIGHT_TOLERANCE).any():
        raise PortfolioConstructionError("weight vector violates long-only bounds")
    arr = np.clip(arr, 0.0, None)
    total = float(arr.sum())
    if not np.isfinite(total) or total <= 0:
        raise PortfolioConstructionError("weight vector has non-positive sum")
    arr = arr / total
    if abs(float(arr.sum()) - 1.0) > WEIGHT_TOLERANCE:
        raise PortfolioConstructionError("weight vector does not sum to one")
    return pd.Series(arr, index=columns, dtype=float)


def _validate_solver_weights(raw_weights: np.ndarray, columns: pd.Index) -> pd.Series:
    """Validate raw optimizer output before repairing tiny floating-point noise."""
    arr = np.asarray(raw_weights, dtype=float).copy()
    if arr.ndim != 1 or arr.shape[0] != len(columns):
        raise PortfolioConstructionError("solver returned wrong weight shape")
    if not np.isfinite(arr).all():
        raise PortfolioConstructionError("solver returned NaN or inf weights")
    if (arr < -WEIGHT_TOLERANCE).any():
        raise PortfolioConstructionError("solver violated long-only bounds")
    if (arr > 1.0 + WEIGHT_TOLERANCE).any():
        raise PortfolioConstructionError("solver violated upper bounds")

    total = float(arr.sum())
    if not np.isclose(total, 1.0, atol=WEIGHT_TOLERANCE, rtol=0.0):
        raise PortfolioConstructionError(
            f"solver weights sum to {total:.12g}, not one"
        )

    arr = np.clip(arr, 0.0, 1.0)
    total = float(arr.sum())
    if not np.isfinite(total) or total <= 0:
        raise PortfolioConstructionError("solver weights have non-positive repaired sum")
    arr = arr / total
    return pd.Series(arr, index=columns, dtype=float)


def _validate_weights(weights: pd.Series, columns: pd.Index) -> pd.Series:
    if not isinstance(weights, pd.Series):
        raise PortfolioConstructionError("weights must be returned as a pandas Series")
    weights = weights.reindex(columns)
    return _normalise_weight_array(weights.to_numpy(dtype=float), columns)


def _return_moments(returns_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    mean = returns_df.mean(axis=0).to_numpy(dtype=float)
    cov = returns_df.cov(ddof=1).to_numpy(dtype=float)
    if not np.isfinite(mean).all() or not np.isfinite(cov).all():
        raise PortfolioConstructionError("return moments contain non-finite values")
    cov = (cov + cov.T) / 2.0
    scale = max(float(np.max(np.abs(cov))), RISK_EPSILON)
    cov = cov / scale
    diag_scale = max(float(np.trace(cov)) / max(cov.shape[0], 1), RISK_EPSILON)
    cov = cov + np.eye(cov.shape[0]) * diag_scale * 1e-8
    return mean, cov


def _inverse_variance_weights(cov: np.ndarray) -> np.ndarray:
    diag = np.diag(cov).astype(float)
    diag = np.where(np.isfinite(diag) & (diag > RISK_EPSILON), diag, np.nan)
    if np.isnan(diag).all():
        return np.repeat(1.0 / len(diag), len(diag))
    inv = 1.0 / np.nan_to_num(diag, nan=np.nanmax(diag))
    inv = np.where(np.isfinite(inv) & (inv > 0), inv, 0.0)
    total = inv.sum()
    if total <= 0:
        return np.repeat(1.0 / len(diag), len(diag))
    return inv / total


def get_equal_weights(returns_df: pd.DataFrame) -> pd.Series:
    """Return equal long-only weights in the same order as ``returns_df.columns``."""
    clean = _prepare_weight_input(returns_df)
    weights = np.repeat(1.0 / clean.shape[1], clean.shape[1])
    return pd.Series(weights, index=clean.columns, dtype=float)


def get_min_variance_weights(returns_df: pd.DataFrame) -> pd.Series:
    """Return long-only fully invested minimum-variance weights."""
    clean = _prepare_weight_input(returns_df)
    n_assets = clean.shape[1]
    if n_assets == 1:
        return pd.Series([1.0], index=clean.columns, dtype=float)

    _, cov = _return_moments(clean)

    def objective(weights: np.ndarray) -> float:
        return float(weights @ cov @ weights)

    x0 = _inverse_variance_weights(cov)
    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_assets,
        constraints=({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},),
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        raise PortfolioConstructionError(f"minimum variance optimizer failed: {result.message}")
    return _validate_solver_weights(result.x, clean.columns)


def get_tangency_weights(returns_df: pd.DataFrame) -> pd.Series:
    """Return long-only maximum-Sharpe tangency weights with risk-free rate zero."""
    clean = _prepare_weight_input(returns_df)
    n_assets = clean.shape[1]
    raw_cov = clean.cov(ddof=1).to_numpy(dtype=float)
    if not np.isfinite(raw_cov).all():
        raise PortfolioConstructionError("maximum Sharpe raw covariance contains non-finite values")
    raw_cov = (raw_cov + raw_cov.T) / 2.0
    mean, cov = _return_moments(clean)

    def objective(weights: np.ndarray) -> float:
        portfolio_return = float(weights @ mean)
        variance = float(weights @ cov @ weights)
        if not np.isfinite(variance) or variance <= RISK_EPSILON:
            return 1e6
        volatility = np.sqrt(variance)
        sharpe = portfolio_return / volatility
        if not np.isfinite(sharpe):
            return 1e6
        return -float(sharpe)

    if n_assets == 1:
        weights = pd.Series([1.0], index=clean.columns, dtype=float)
    else:
        x0 = _inverse_variance_weights(cov)
        result = minimize(
            objective,
            x0=x0,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n_assets,
            constraints=({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},),
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if not result.success:
            raise PortfolioConstructionError(f"maximum Sharpe optimizer failed: {result.message}")
        weights = _validate_solver_weights(result.x, clean.columns)

    weight_values = weights.to_numpy(dtype=float)
    raw_variance = float(weight_values @ raw_cov @ weight_values)
    if not np.isfinite(raw_variance) or raw_variance <= RISK_EPSILON:
        raise PortfolioConstructionError(
            "maximum Sharpe solution has near-zero raw variance"
        )
    objective_value = objective(weight_values)
    if not np.isfinite(objective_value) or objective_value >= 1e6:
        raise PortfolioConstructionError(
            "maximum Sharpe optimizer returned an invalid objective"
        )
    return weights


def _cluster_variance(cov: pd.DataFrame, tickers: list[str]) -> float:
    sub_cov = cov.loc[tickers, tickers].to_numpy(dtype=float)
    weights = _inverse_variance_weights(sub_cov)
    variance = float(weights @ sub_cov @ weights)
    if not np.isfinite(variance) or variance < 0:
        raise PortfolioConstructionError("invalid HRP cluster variance")
    return max(variance, RISK_EPSILON)


def get_hrp_weights(returns_df: pd.DataFrame) -> pd.Series:
    """Return Hierarchical Risk Parity weights.

    The implementation uses correlation distance, SciPy hierarchical linkage,
    leaf ordering, quasi-diagonalisation, and recursive bisection allocation.
    """
    clean = _prepare_weight_input(returns_df)
    n_assets = clean.shape[1]
    if n_assets == 1:
        return pd.Series([1.0], index=clean.columns, dtype=float)

    cov = clean.cov(ddof=1)
    corr = clean.corr()
    if cov.isna().any().any() or corr.isna().any().any():
        raise PortfolioConstructionError("HRP covariance or correlation contains NaN")
    if not np.isfinite(cov.to_numpy(dtype=float)).all() or not np.isfinite(corr.to_numpy(dtype=float)).all():
        raise PortfolioConstructionError("HRP covariance or correlation contains non-finite values")

    corr_array = corr.clip(-1.0, 1.0).to_numpy(dtype=float, copy=True)
    np.fill_diagonal(corr_array, 1.0)
    corr = pd.DataFrame(corr_array, index=clean.columns, columns=clean.columns)
    distance = np.sqrt((1.0 - corr) / 2.0)
    if not np.isfinite(distance.to_numpy(dtype=float)).all():
        raise PortfolioConstructionError("HRP correlation distance contains invalid values")

    condensed = squareform(distance.to_numpy(dtype=float), checks=False)
    if not np.isfinite(condensed).all():
        raise PortfolioConstructionError("HRP condensed distance contains invalid values")

    try:
        link = linkage(condensed, method="single")
        order = leaves_list(link)
    except Exception as exc:  # pragma: no cover - SciPy raises several exception types.
        raise PortfolioConstructionError(f"HRP linkage failed: {exc}") from exc
    if len(order) != n_assets:
        raise PortfolioConstructionError("HRP leaf ordering has the wrong length")

    ordered_tickers = list(clean.columns[order])
    weights = pd.Series(1.0, index=ordered_tickers, dtype=float)
    clusters: list[list[str]] = [ordered_tickers]
    while clusters:
        next_clusters: list[list[str]] = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            split = len(cluster) // 2
            left = cluster[:split]
            right = cluster[split:]
            left_var = _cluster_variance(cov, left)
            right_var = _cluster_variance(cov, right)
            alpha = 1.0 - left_var / (left_var + right_var)
            if not np.isfinite(alpha):
                raise PortfolioConstructionError("HRP recursive allocation produced invalid alpha")
            weights.loc[left] *= alpha
            weights.loc[right] *= 1.0 - alpha
            next_clusters.extend([left, right])
        clusters = next_clusters

    weights = weights.reindex(clean.columns)
    return _normalise_weight_array(weights.to_numpy(dtype=float), clean.columns)


def _method_function(method: str) -> Callable[[pd.DataFrame], pd.Series]:
    normalized = _normalise_method(method)
    return {
        "equal_weight": get_equal_weights,
        "minimum_variance": get_min_variance_weights,
        "maximum_sharpe": get_tangency_weights,
        "hrp": get_hrp_weights,
    }[normalized]


def _monthly_formation_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    dates = pd.Series(index=index, data=index)
    return pd.DatetimeIndex(dates.groupby(index.to_period("M")).tail(1).to_numpy())


def _eligible_assets(estimation_sample: pd.DataFrame) -> list[str]:
    """Determine investable assets using formation-date information only."""
    complete_estimation = estimation_sample.notna().all(axis=0)
    finite_estimation = np.isfinite(estimation_sample.to_numpy(dtype=float)).all(axis=0)
    mask = complete_estimation & pd.Series(finite_estimation, index=estimation_sample.columns)
    return list(estimation_sample.columns[mask.to_numpy()])


def _build_target_schedule(
    returns_df: pd.DataFrame,
    method: str,
    window: int,
) -> tuple[dict[pd.Timestamp, pd.Series], list[dict[str, object]], list[dict[str, object]]]:
    method_key = _normalise_method(method)
    method_func = _method_function(method_key)
    all_dates = returns_df.index
    positions = {date: i for i, date in enumerate(all_dates)}
    targets: dict[pd.Timestamp, pd.Series] = {}
    weight_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []

    for formation_date in _monthly_formation_dates(all_dates):
        pos = positions[pd.Timestamp(formation_date)]
        if pos < window - 1 or pos + 1 >= len(all_dates):
            continue
        effective_date = all_dates[pos + 1]
        estimation_sample = returns_df.iloc[pos - window + 1 : pos + 1]
        eligible = _eligible_assets(estimation_sample)
        solver_success = True
        fallback_used = False
        message = "success"

        if not eligible:
            solver_success = False
            fallback_used = True
            message = "no eligible assets with complete finite estimation returns"
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            diagnostic_rows.append(
                {
                    "formation_date": pd.Timestamp(formation_date),
                    "effective_date": pd.Timestamp(effective_date),
                    "method": method_key,
                    "n_assets": 0,
                    "solver_success": solver_success,
                    "fallback_used": fallback_used,
                    "message": message,
                    "turnover": np.nan,
                    "transaction_cost": np.nan,
                }
            )
            continue

        method_sample = estimation_sample.loc[:, eligible]
        try:
            target = method_func(method_sample)
            target = _validate_weights(target, method_sample.columns)
        except Exception as exc:
            solver_success = False
            fallback_used = True
            message = f"fallback to equal weight: {exc}"
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            target = pd.Series(
                np.repeat(1.0 / len(eligible), len(eligible)),
                index=pd.Index(eligible),
                dtype=float,
            )

        full_target = pd.Series(0.0, index=returns_df.columns, dtype=float)
        full_target.loc[target.index] = target
        full_target = _normalise_weight_array(full_target.to_numpy(dtype=float), returns_df.columns)
        targets[pd.Timestamp(effective_date)] = full_target

        for ticker, weight in full_target.items():
            weight_rows.append(
                {
                    "formation_date": pd.Timestamp(formation_date),
                    "effective_date": pd.Timestamp(effective_date),
                    "ticker": ticker,
                    "target_weight": float(weight),
                }
            )
        diagnostic_rows.append(
            {
                "formation_date": pd.Timestamp(formation_date),
                "effective_date": pd.Timestamp(effective_date),
                "method": method_key,
                "n_assets": len(eligible),
                "solver_success": solver_success,
                "fallback_used": fallback_used,
                "message": message,
                "turnover": np.nan,
                "transaction_cost": np.nan,
            }
        )

    return targets, weight_rows, diagnostic_rows


def _asset_returns_for_day(returns_row: pd.Series, active_weights: pd.Series) -> pd.Series:
    aligned = returns_row.reindex(active_weights.index).astype(float)
    missing_active = active_weights.gt(0.0) & ~np.isfinite(aligned)
    if missing_active.any():
        tickers = list(active_weights.index[missing_active])
        raise ValueError(
            "missing holding-period returns for active assets: "
            f"{tickers[:5]}"
        )
    return aligned.fillna(0.0)


def _update_diagnostic_costs(
    diagnostics: list[dict[str, object]],
    effective_date: pd.Timestamp,
    turnover: float,
    transaction_cost: float,
) -> None:
    for row in diagnostics:
        if pd.Timestamp(row["effective_date"]) == effective_date:
            row["turnover"] = float(turnover)
            row["transaction_cost"] = float(transaction_cost)
            return


def _empty_execution_diagnostics_frame() -> pd.DataFrame:
    """Execution-side diagnostics recorded by the shared backtest engine.

    Deliberately narrower than ``_empty_diagnostics_frame``: the executor does
    not know how a target vector was produced, so it reports only what it
    observed while trading the schedule.
    """
    return pd.DataFrame(
        columns=["formation_date", "effective_date", "turnover", "transaction_cost"]
    )


def run_backtest_from_target_schedule(
    returns_df: pd.DataFrame,
    targets_by_effective_date: Mapping[pd.Timestamp, pd.Series],
    formation_date_by_effective_date: Mapping[pd.Timestamp, pd.Timestamp],
    transaction_cost_rate: float = 0.001,
) -> BacktestResult:
    """Trade a pre-built target-weight schedule and report the result.

    This is the single execution engine behind both the four portfolio methods
    and the sentiment-tilt fusion variants, so every fund in the project shares
    one rebalance timing, weight-drift, turnover and transaction-cost
    definition and the results stay directly comparable.

    ``targets_by_effective_date`` is keyed by EFFECTIVE date - the first date a
    target vector is actually held. ``formation_date_by_effective_date`` maps
    ``effective_date -> formation_date`` and is recorded as audit metadata
    only; it never affects when a target starts to apply. The two are separate
    arguments rather than one ambiguously-named mapping so a caller cannot
    quietly pass the relationship the wrong way round.
    """
    if not np.isfinite(transaction_cost_rate) or not 0.0 <= transaction_cost_rate < 1.0:
        raise ValueError("transaction_cost_rate must be finite and in [0, 1)")

    clean = _validate_returns_frame(returns_df)
    targets = _validate_target_schedule(clean, targets_by_effective_date)

    weight_rows: list[dict[str, object]] = []
    for effective_date, target in targets.items():
        if effective_date not in formation_date_by_effective_date:
            raise ValueError(
                f"no formation_date recorded for effective_date {effective_date}"
            )
        formation_date = pd.Timestamp(formation_date_by_effective_date[effective_date])
        if formation_date >= effective_date:
            raise ValueError(
                f"formation_date {formation_date} must be strictly before "
                f"effective_date {effective_date}; the mapping direction is "
                "effective_date -> formation_date"
            )
        for ticker, weight in target.items():
            weight_rows.append(
                {
                    "formation_date": formation_date,
                    "effective_date": effective_date,
                    "ticker": ticker,
                    "target_weight": float(weight),
                }
            )
    weights = pd.DataFrame(weight_rows, columns=_empty_weights_frame().columns)

    if not targets:
        return BacktestResult(
            returns=_empty_returns_frame(),
            weights=weights,
            diagnostics=_empty_execution_diagnostics_frame(),
        )

    first_effective = min(targets)
    current_weights = pd.Series(0.0, index=clean.columns, dtype=float)
    has_position = False
    return_rows: list[dict[str, object]] = []
    execution_rows: list[dict[str, object]] = []

    for date in clean.index[clean.index >= first_effective]:
        date = pd.Timestamp(date)
        start_weights = current_weights.copy()
        transaction_cost = 0.0
        turnover = 0.0

        if date in targets:
            target = targets[date]
            if has_position:
                turnover = 0.5 * float(np.abs(target - start_weights).sum())
            else:
                turnover = 1.0
            transaction_cost = transaction_cost_rate * turnover
            start_weights = target.copy()
            has_position = True
            execution_rows.append(
                {
                    "formation_date": pd.Timestamp(
                        formation_date_by_effective_date[date]
                    ),
                    "effective_date": date,
                    "turnover": float(turnover),
                    "transaction_cost": float(transaction_cost),
                }
            )

        asset_returns = _asset_returns_for_day(clean.loc[date], start_weights)
        gross_return = float(start_weights @ asset_returns)
        net_return = gross_return - transaction_cost
        if not np.isfinite(net_return) or net_return <= -1.0:
            raise ValueError(f"net return is invalid on {date}: {net_return}")
        return_rows.append(
            {
                "date": date,
                "gross_return": gross_return,
                "transaction_cost": transaction_cost,
                "net_return": net_return,
            }
        )

        gross_growth = 1.0 + gross_return
        if has_position and np.isfinite(gross_growth) and gross_growth > RISK_EPSILON:
            current_weights = start_weights * (1.0 + asset_returns)
            total = float(current_weights.sum())
            if np.isfinite(total) and total > RISK_EPSILON:
                current_weights = current_weights / total
            else:
                current_weights = pd.Series(0.0, index=clean.columns, dtype=float)
                has_position = False
        else:
            current_weights = pd.Series(0.0, index=clean.columns, dtype=float)
            has_position = False

    returns = pd.DataFrame(return_rows).set_index("date")
    diagnostics = pd.DataFrame(
        execution_rows, columns=_empty_execution_diagnostics_frame().columns
    )
    return BacktestResult(returns=returns, weights=weights, diagnostics=diagnostics)


def _validate_target_schedule(
    clean: pd.DataFrame,
    targets_by_effective_date: Mapping[pd.Timestamp, pd.Series],
) -> dict[pd.Timestamp, pd.Series]:
    """Validate a caller-supplied ``effective_date -> weights`` map.

    A target naming a ticker the return panel does not contain is rejected
    rather than reindexed away: dropping it would silently break the
    "weights sum to one" invariant the executor depends on.
    """
    validated: dict[pd.Timestamp, pd.Series] = {}
    for raw_date, raw_target in targets_by_effective_date.items():
        effective_date = pd.Timestamp(raw_date)
        if effective_date not in clean.index:
            raise ValueError(
                f"target effective_date {effective_date} is not a date in returns_df"
            )
        if not isinstance(raw_target, pd.Series):
            raise TypeError(f"target for {effective_date} must be a pandas Series")
        if raw_target.index.has_duplicates:
            raise ValueError(f"target for {effective_date} has duplicate tickers")
        unknown = set(raw_target.index) - set(clean.columns)
        if unknown:
            raise ValueError(
                f"target for {effective_date} references tickers absent from "
                f"returns_df: {sorted(unknown)[:5]}"
            )
        full_target = pd.Series(0.0, index=clean.columns, dtype=float)
        full_target.loc[raw_target.index] = raw_target.astype(float)
        values = full_target.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"target for {effective_date} contains NaN or inf")
        if (values < -WEIGHT_TOLERANCE).any():
            raise ValueError(f"target for {effective_date} violates long-only bounds")
        if abs(float(values.sum()) - 1.0) > WEIGHT_TOLERANCE:
            raise ValueError(
                f"target for {effective_date} sums to {values.sum():.12g}, not one"
            )
        validated[effective_date] = full_target
    return dict(sorted(validated.items()))


def run_walk_forward_backtest(
    returns_df: pd.DataFrame,
    method: str,
    window: int,
    annualization_factor: int,
    transaction_cost_rate: float = 0.001,
) -> BacktestResult:
    """Run a monthly walk-forward out-of-sample backtest.

    Weights are formed on each calendar month's last available observation date
    using the previous ``window`` observations through that formation date. The
    new target is first applied on the next available date, so a formation-date
    signal never affects the formation-date return. Between rebalances, weights
    drift buy-and-hold with asset returns.

    This function now only turns a method name into a target schedule; the
    schedule is priced by the shared ``run_backtest_from_target_schedule``
    executor, so the methods and the fusion variants cannot drift apart.
    """
    method_key = _normalise_method(method)
    window = _validate_positive_integer(window, "window")
    annualization_factor = _validate_positive_integer(
        annualization_factor, "annualization_factor"
    )
    if window < 2:
        raise ValueError("window must be at least 2")
    if not np.isfinite(transaction_cost_rate) or not 0.0 <= transaction_cost_rate < 1.0:
        raise ValueError("transaction_cost_rate must be finite and in [0, 1)")

    clean = _validate_returns_frame(returns_df)
    targets, weight_rows, diagnostic_rows = _build_target_schedule(clean, method_key, window)
    if not targets:
        return BacktestResult(
            returns=_empty_returns_frame(),
            weights=pd.DataFrame(weight_rows, columns=_empty_weights_frame().columns),
            diagnostics=pd.DataFrame(diagnostic_rows, columns=_empty_diagnostics_frame().columns),
        )

    formation_by_effective = {
        pd.Timestamp(row["effective_date"]): pd.Timestamp(row["formation_date"])
        for row in diagnostic_rows
    }
    executed = run_backtest_from_target_schedule(
        clean,
        targets,
        formation_by_effective,
        transaction_cost_rate=transaction_cost_rate,
    )

    # Fold the executor's realised turnover and cost back into the
    # method-level diagnostic rows. A formation date that produced no target
    # (for example one with no eligible assets) keeps its NaN placeholders.
    for record in executed.diagnostics.to_dict("records"):
        _update_diagnostic_costs(
            diagnostic_rows,
            pd.Timestamp(record["effective_date"]),
            float(record["turnover"]),
            float(record["transaction_cost"]),
        )

    diagnostics = pd.DataFrame(diagnostic_rows, columns=_empty_diagnostics_frame().columns)
    return BacktestResult(
        returns=executed.returns, weights=executed.weights, diagnostics=diagnostics
    )


def calculate_growth_of_1(returns_series: pd.Series) -> pd.Series:
    """Return cumulative growth of 1 from a net-return series."""
    if not isinstance(returns_series, pd.Series):
        raise TypeError("returns_series must be a pandas Series")
    clean = pd.to_numeric(returns_series, errors="coerce").dropna()
    if clean.empty:
        return pd.Series(dtype=float, name="growth_of_1")
    if np.isinf(clean.to_numpy(dtype=float)).any():
        raise ValueError("returns_series contains inf or -inf")
    if (clean < -1.0).any():
        raise ValueError("returns_series contains returns below -100%")
    growth = (1.0 + clean).cumprod()
    growth.name = "growth_of_1"
    return growth


def _max_drawdown_from_returns(returns: pd.Series) -> float:
    if returns.empty:
        return np.nan
    wealth = calculate_growth_of_1(returns)
    if wealth.empty:
        return np.nan
    drawdown = _drawdown_from_growth(wealth)
    return float(drawdown.min())


def calculate_performance_metrics(
    returns_series: pd.Series,
    annualization_factor: int,
) -> dict[str, float]:
    """Calculate geometric return, volatility, Sharpe ratio, and max drawdown."""
    annualization_factor = _validate_positive_integer(
        annualization_factor, "annualization_factor"
    )
    if not isinstance(returns_series, pd.Series):
        raise TypeError("returns_series must be a pandas Series")

    clean = pd.to_numeric(returns_series, errors="coerce").dropna()
    metrics = {
        "annual_return": np.nan,
        "annual_volatility": np.nan,
        "sharpe_ratio": np.nan,
        "max_drawdown": np.nan,
    }
    if clean.empty:
        return metrics
    if np.isinf(clean.to_numpy(dtype=float)).any():
        raise ValueError("returns_series contains inf or -inf")
    if (clean < -1.0).any():
        raise ValueError("returns_series contains returns below -100%")

    n_obs = int(clean.shape[0])
    wealth_end = float((1.0 + clean).prod())
    if wealth_end > 0:
        metrics["annual_return"] = wealth_end ** (annualization_factor / n_obs) - 1.0
    elif wealth_end == 0:
        metrics["annual_return"] = -1.0

    if n_obs >= 2:
        vol = float(clean.std(ddof=1) * np.sqrt(annualization_factor))
        metrics["annual_volatility"] = vol
        if vol > RISK_EPSILON:
            metrics["sharpe_ratio"] = float(clean.mean() / clean.std(ddof=1) * np.sqrt(annualization_factor))
    metrics["max_drawdown"] = _max_drawdown_from_returns(clean)
    return metrics


def extract_current_holdings(weights_df: pd.DataFrame) -> pd.DataFrame:
    """Extract latest formation-date target holdings sorted by weight descending."""
    required = {"formation_date", "ticker", "target_weight"}
    missing = required - set(weights_df.columns)
    if missing:
        raise ValueError(f"weights_df is missing required columns: {sorted(missing)}")
    if weights_df.empty:
        return pd.DataFrame(columns=list(weights_df.columns))
    df = weights_df.copy()
    df["formation_date"] = pd.to_datetime(df["formation_date"])
    df["target_weight"] = pd.to_numeric(df["target_weight"], errors="coerce")
    latest = df["formation_date"].max()
    return (
        df.loc[df["formation_date"].eq(latest)]
        .sort_values("target_weight", ascending=False)
        .reset_index(drop=True)
    )


def _drawdown_from_growth(growth: pd.Series) -> pd.Series:
    if growth.empty:
        return pd.Series(dtype=float, index=growth.index, name="drawdown")
    values = growth.to_numpy(dtype=float)
    running_high = np.maximum.accumulate(np.concatenate(([1.0], values)))[1:]
    return pd.Series(values / running_high - 1.0, index=growth.index, name="drawdown")


def _standardise_returns(
    result: BacktestResult,
    fund_family: str,
    method: str,
    fund_name: str,
) -> pd.DataFrame:
    if result.returns.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "fund_family",
                "method",
                "fund_name",
                "gross_return",
                "transaction_cost",
                "net_return",
                "growth_of_1",
                "drawdown",
            ]
        )
    out = result.returns.copy()
    growth = calculate_growth_of_1(out["net_return"])
    out["growth_of_1"] = growth.reindex(out.index)
    out["drawdown"] = _drawdown_from_growth(out["growth_of_1"])
    out = out.reset_index()
    out.insert(1, "fund_family", fund_family)
    out.insert(2, "method", method)
    out.insert(3, "fund_name", fund_name)
    return out[
        [
            "date",
            "fund_family",
            "method",
            "fund_name",
            "gross_return",
            "transaction_cost",
            "net_return",
            "growth_of_1",
            "drawdown",
        ]
    ]


def _standardise_weights(
    result: BacktestResult,
    fund_family: str,
    method: str,
    fund_name: str,
) -> pd.DataFrame:
    out = result.weights.copy()
    if out.empty:
        out = _empty_weights_frame()
    out.insert(2, "fund_family", fund_family)
    out.insert(3, "method", method)
    out.insert(4, "fund_name", fund_name)
    return out[
        [
            "formation_date",
            "effective_date",
            "fund_family",
            "method",
            "fund_name",
            "ticker",
            "target_weight",
        ]
    ]


def _standardise_diagnostics(
    result: BacktestResult,
    fund_family: str,
    method: str,
    fund_name: str,
) -> pd.DataFrame:
    out = result.diagnostics.copy()
    if out.empty:
        out = _empty_diagnostics_frame()
    out.insert(2, "fund_family", fund_family)
    out.insert(4, "fund_name", fund_name)
    return out[
        [
            "formation_date",
            "effective_date",
            "fund_family",
            "method",
            "fund_name",
            "n_assets",
            "solver_success",
            "fallback_used",
            "message",
            "turnover",
            "transaction_cost",
        ]
    ]


def _metrics_row(
    returns: pd.DataFrame,
    fund_family: str,
    method: str,
    fund_name: str,
    annualization_factor: int,
) -> dict[str, object]:
    metrics = calculate_performance_metrics(returns["net_return"], annualization_factor)
    return {
        "fund_name": fund_name,
        "fund_family": fund_family,
        "method": method,
        "annualization_factor": annualization_factor,
        "oos_start": returns.index.min() if not returns.empty else pd.NaT,
        "oos_end": returns.index.max() if not returns.empty else pd.NaT,
        **metrics,
    }


def _annotate_similar_method_weights(weights: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    if weights.empty or diagnostics.empty:
        return diagnostics
    out = diagnostics.copy()
    key_cols = ["fund_family", "formation_date"]
    for (_, formation_date), group in weights.groupby(key_cols):
        pivot = group.pivot_table(
            index="ticker", columns="method", values="target_weight", fill_value=0.0
        )
        methods = list(pivot.columns)
        for i, left in enumerate(methods):
            for right in methods[i + 1 :]:
                diff = float(np.abs(pivot[left] - pivot[right]).max())
                if diff <= 1e-8:
                    mask = out["formation_date"].eq(formation_date) & out["method"].isin([left, right])
                    note = f"weights nearly identical to another method at {pd.Timestamp(formation_date).date()}"
                    out.loc[mask, "message"] = out.loc[mask, "message"].astype(str) + f"; {note}"
    return out


def build_fund_universe(
    equity_returns: pd.DataFrame,
    crypto_returns: pd.DataFrame,
    combined_returns: pd.DataFrame,
    methods: Sequence[str] = METHODS,
    transaction_cost_rate: float = 0.001,
) -> dict[str, pd.DataFrame]:
    """Run all requested fund-family and method backtests.

    Parameters are expected to be wide date-by-ticker return panels. The returned
    dictionary contains DataFrames ready for ``scripts/run_part_b.py`` to save.
    """
    panels = {
        "Equity-Only": _validate_returns_frame(equity_returns),
        "Crypto-Only": _validate_returns_frame(crypto_returns),
        "Combined": _validate_returns_frame(combined_returns),
    }
    method_keys = tuple(_normalise_method(method) for method in methods)

    all_returns: list[pd.DataFrame] = []
    all_weights: list[pd.DataFrame] = []
    all_diagnostics: list[pd.DataFrame] = []
    all_metrics: list[dict[str, object]] = []

    for fund_family, panel in panels.items():
        config = FAMILY_CONFIG[fund_family]
        window = int(config["window"])
        annualization_factor = int(config["annualization_factor"])
        for method in method_keys:
            fund_name = f"{fund_family} {METHOD_LABELS[method]}"
            result = run_walk_forward_backtest(
                panel,
                method=method,
                window=window,
                annualization_factor=annualization_factor,
                transaction_cost_rate=transaction_cost_rate,
            )
            returns_out = _standardise_returns(result, fund_family, method, fund_name)
            weights_out = _standardise_weights(result, fund_family, method, fund_name)
            diagnostics_out = _standardise_diagnostics(result, fund_family, method, fund_name)
            all_returns.append(returns_out)
            all_weights.append(weights_out)
            all_diagnostics.append(diagnostics_out)
            all_metrics.append(
                _metrics_row(result.returns, fund_family, method, fund_name, annualization_factor)
            )

    fund_returns = pd.concat(all_returns, ignore_index=True)
    fund_weights = pd.concat(all_weights, ignore_index=True)
    diagnostics = pd.concat(all_diagnostics, ignore_index=True)
    diagnostics = _annotate_similar_method_weights(fund_weights, diagnostics)
    performance_metrics = pd.DataFrame(all_metrics)

    return {
        "fund_returns": fund_returns[
            [
                "date",
                "fund_family",
                "method",
                "fund_name",
                "gross_return",
                "transaction_cost",
                "net_return",
                "growth_of_1",
                "drawdown",
            ]
        ],
        "fund_weights": fund_weights[
            [
                "formation_date",
                "effective_date",
                "fund_family",
                "method",
                "fund_name",
                "ticker",
                "target_weight",
            ]
        ],
        "performance_metrics": performance_metrics[
            [
                "fund_name",
                "fund_family",
                "method",
                "annualization_factor",
                "oos_start",
                "oos_end",
                "annual_return",
                "annual_volatility",
                "sharpe_ratio",
                "max_drawdown",
            ]
        ],
        "diagnostics": diagnostics[
            [
                "formation_date",
                "effective_date",
                "fund_family",
                "method",
                "fund_name",
                "n_assets",
                "solver_success",
                "fallback_used",
                "message",
                "turnover",
                "transaction_cost",
            ]
        ],
    }
