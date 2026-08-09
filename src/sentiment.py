"""FinVADER baseline scoring, 2020 candidate-term discovery, and the extended
(baseline + RiskBridge custom lexicon) sentiment build for 2021-2023."""
from __future__ import annotations

import importlib
import importlib.metadata as metadata
import math
import os
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from . import sentiment_lexicon

FINVADER_INDICATOR = "compound"
FINVADER_USE_SENTIBIGNOMICS = True
FINVADER_USE_HENRY = True
DISCOVERY_START_DATE = "2020-01-01"
DISCOVERY_END_DATE = "2020-12-31"
APPLICATION_START_DATE = "2021-01-01"
APPLICATION_END_DATE = "2023-12-31"
TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z'-]{2,}")
CANDIDATE_COLUMNS = [
    "term",
    "frequency",
    "document_frequency",
    "in_lexicon",
    "current_valence",
    "n_tickers",
    "n_sectors",
    "example_headlines",
    "window_start",
    "window_end",
    "n_headlines_in_window",
    "is_stopword",
    "review_eligible",
    "exclusion_reason",
]

# Diagnostic only: these terms remain in the complete candidate table.
REVIEW_STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "from", "that", "this", "into", "over",
        "under", "after", "before", "amid", "will", "says", "say", "said",
        "new", "its", "are", "has", "have", "had", "was", "were", "been",
        "more", "less", "than", "about", "how", "why", "what", "when",
        "where", "your", "you", "our", "their", "his", "her", "not", "but",
        "out", "off", "all", "can", "may", "could", "would", "should",
        "stock", "stocks", "market", "markets", "company", "companies",
        "shares", "share", "inc", "corp", "ltd", "plc", "nasdaq", "nyse",
        "us",
    }
)


def _require_local_vader_lexicon(nltk_module: object) -> None:
    resource_paths = (
        "sentiment/vader_lexicon.zip/vader_lexicon/vader_lexicon.txt",
        "sentiment/vader_lexicon.zip",
        "sentiment/vader_lexicon",
    )
    for resource_path in resource_paths:
        try:
            nltk_module.data.find(resource_path)
            return
        except LookupError:
            continue
    raise LookupError(
        "NLTK vader_lexicon is not installed locally. Install it once in a "
        "controlled setup step before running sentiment scoring."
    )


def load_finvader_module() -> object:
    """Import FinVADER while preventing its import-time NLTK download."""
    import nltk

    _require_local_vader_lexicon(nltk)
    original_download = nltk.download

    def blocked_download(*args: object, **kwargs: object) -> bool:
        return True

    # FinVADER 1.0.2 downloads at import time; local resources are required above.
    try:
        nltk.download = blocked_download
        return importlib.import_module("finvader")
    finally:
        nltk.download = original_download


def load_finvader_scorer() -> Callable[..., float]:
    """Return the installed package's public ``finvader`` scoring function."""
    scorer = getattr(load_finvader_module(), "finvader", None)
    if not callable(scorer):
        raise ImportError("The installed finvader package has no public finvader function")
    return scorer


def finvader_metadata() -> dict[str, object]:
    """Return the installed version and fixed baseline scoring settings."""
    return {
        "finvader_version": metadata.version("finvader"),
        "indicator": FINVADER_INDICATOR,
        "use_sentibignomics": FINVADER_USE_SENTIBIGNOMICS,
        "use_henry": FINVADER_USE_HENRY,
    }


def score_headlines(
    panel: pd.DataFrame,
    *,
    text_col: str = "raw_title",
    scorer: Callable[..., float] | None = None,
) -> pd.DataFrame:
    """Score every unmodified raw title with FinVADER's public function."""
    if text_col not in panel.columns:
        raise ValueError(f"headline panel is missing required column: {text_col!r}")

    titles = panel[text_col]
    invalid = titles.map(lambda value: not isinstance(value, str) or not value.strip())
    if invalid.any():
        raise ValueError(
            f"headline panel contains {int(invalid.sum())} missing, blank, or "
            f"non-string value(s) in {text_col!r}"
        )

    public_scorer = scorer or load_finvader_scorer()
    scores: list[float] = []
    for raw_title in titles.tolist():
        score = public_scorer(
            raw_title,
            indicator=FINVADER_INDICATOR,
            use_sentibignomics=FINVADER_USE_SENTIBIGNOMICS,
            use_henry=FINVADER_USE_HENRY,
        )
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("FinVADER returned a non-finite or non-numeric score")
        numeric_score = float(score)
        if not math.isfinite(numeric_score):
            raise ValueError("FinVADER returned a non-finite or non-numeric score")
        scores.append(numeric_score)

    result = panel.copy()
    result["sentiment_score"] = scores
    return result


def effective_finvader_lexicon(
    *,
    module: object | None = None,
    analyzer_factory: Callable[[], object] | None = None,
    custom_lexicon: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Build the effective VADER, SentiBignomics, then Henry lexicon.

    Merge order matches FinVADER's own ``finvader()`` with
    ``use_sentibignomics=True, use_henry=True`` (verified against the public
    scorer in tests/test_sentiment.py): native VADER, then SentiBignomics
    scaled by 0.1, then Henry raw.

    If ``custom_lexicon`` is given, its terms are merged as one more layer on
    top, RAW - never scaled by 0.1 (that scaling is SentiBignomics-specific).
    Overlap between ``custom_lexicon`` and the native+SentiBignomics+Henry
    lexicon built so far raises ValueError naming the overlapping terms,
    rather than silently overwriting them.
    """
    if module is None:
        module = load_finvader_module()
    if analyzer_factory is None:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        analyzer_factory = SentimentIntensityAnalyzer

    analyzer = analyzer_factory()
    effective = {
        str(term).lower(): float(value)
        for term, value in analyzer.lexicon.items()
    }
    sentibignomics = {
        str(term).lower(): float(value) * 0.1
        for term, value in module.lexicon1().items()
    }
    henry = {
        str(term).lower(): float(value)
        for term, value in module.lexicon2().items()
    }
    effective.update(sentibignomics)
    effective.update(henry)
    if not all(math.isfinite(value) for value in effective.values()):
        raise ValueError("effective FinVADER lexicon contains a non-finite valence")

    if custom_lexicon is not None:
        custom = {str(term).lower(): float(value) for term, value in custom_lexicon.items()}
        if not all(math.isfinite(value) for value in custom.values()):
            raise ValueError("custom_lexicon contains a non-finite valence")
        overlap = sorted(set(custom) & set(effective))
        if overlap:
            raise ValueError(
                "custom_lexicon terms overlap the baseline effective FinVADER "
                "lexicon (native VADER + SentiBignomics x0.1 + Henry) and would "
                f"silently overwrite an existing entry: {overlap}"
            )
        effective.update(custom)  # raw valence - never scaled by 0.1

    return effective


def build_analyzer(
    *,
    custom_lexicon: Mapping[str, float] | None = None,
    module: object | None = None,
) -> object:
    """Build an NLTK VADER analyzer using the effective FinVADER lexicon.

    With ``custom_lexicon=None`` this reproduces the exact lexicon FinVADER's
    public ``finvader()`` scorer uses internally for
    ``use_sentibignomics=True, use_henry=True`` (the baseline model). Passing
    ``custom_lexicon`` adds it as the extended model's final merge layer.
    """
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    analyzer.lexicon = effective_finvader_lexicon(module=module, custom_lexicon=custom_lexicon)
    return analyzer


def vader_lookup_tokens(text: str) -> tuple[str, ...]:
    """Return the lowercase tokens NLTK VADER would use for lexicon lookup."""
    from nltk.sentiment.vader import SentiText, VaderConstants

    constants = VaderConstants()
    senti_text = SentiText(
        text,
        constants.PUNC_LIST,
        constants.REGEX_REMOVE_PUNCTUATION,
    )
    return tuple(token.lower() for token in senti_text.words_and_emoticons)


def token_compatibility_diagnostic(surface_text: str) -> dict[str, object]:
    """Compare candidate-regex terms with VADER's actual lookup tokens."""
    candidate_tokens = tuple(
        match.group(0).lower() for match in TOKEN_PATTERN.finditer(surface_text)
    )
    lookup_tokens = vader_lookup_tokens(surface_text)
    return {
        "surface_text": surface_text,
        "candidate_tokens": candidate_tokens,
        "vader_lookup_tokens": lookup_tokens,
        "exact_match": candidate_tokens == lookup_tokens,
    }


def _review_diagnostics(term: str) -> tuple[bool, bool, str]:
    is_stopword = term in REVIEW_STOPWORDS
    reasons: list[str] = []
    if is_stopword:
        reasons.append("stopword")
    if term.endswith(("-", "'")):
        reasons.append("trailing hyphen or apostrophe")
    if vader_lookup_tokens(term) != (term,):
        reasons.append("VADER token mismatch")
    return is_stopword, not reasons, "; ".join(reasons)


def _empty_candidate_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=CANDIDATE_COLUMNS)


def candidate_terms(
    headlines: pd.DataFrame,
    *,
    date_col: str = "date",
    text_col: str = "raw_title",
    start_date: str = DISCOVERY_START_DATE,
    end_date: str = DISCOVERY_END_DATE,
    top_n: int | None = None,
    baseline_lexicon: Mapping[str, float] | None = None,
    examples_per_term: int = 3,
) -> pd.DataFrame:
    """Count candidate tokens in raw headline rows inside a closed date window."""
    required = {date_col, text_col, "ticker", "sector"}
    missing = required - set(headlines.columns)
    if missing:
        raise ValueError(f"headline panel is missing columns: {sorted(missing)}")
    if top_n is not None and (
        isinstance(top_n, bool) or not isinstance(top_n, int) or top_n <= 0
    ):
        raise ValueError("top_n must be a positive integer or None")
    if (
        isinstance(examples_per_term, bool)
        or not isinstance(examples_per_term, int)
        or examples_per_term <= 0
    ):
        raise ValueError("examples_per_term must be a positive integer")

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if start > end:
        raise ValueError("start_date must be on or before end_date")

    dates = pd.to_datetime(headlines[date_col], errors="coerce")
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_convert(None)
    dates = dates.dt.normalize()
    in_window = dates.between(start, end, inclusive="both")
    discovery = headlines.loc[in_window].copy()
    discovery["_candidate_date"] = dates.loc[in_window].to_numpy()
    if discovery.empty:
        return _empty_candidate_frame()

    invalid_titles = discovery[text_col].map(
        lambda value: not isinstance(value, str) or not value.strip()
    )
    if invalid_titles.any():
        raise ValueError(
            f"discovery period contains {int(invalid_titles.sum())} missing, blank, "
            f"or non-string value(s) in {text_col!r}"
        )

    discovery = discovery.sort_values(
        ["_candidate_date", "ticker", "sector", text_col], kind="mergesort"
    )
    lexicon_source = (
        baseline_lexicon
        if baseline_lexicon is not None
        else effective_finvader_lexicon()
    )
    lexicon = {str(term).lower(): float(value) for term, value in lexicon_source.items()}
    if not all(math.isfinite(value) for value in lexicon.values()):
        raise ValueError("baseline_lexicon contains a non-finite valence")

    records: dict[str, dict[str, object]] = {}
    for raw_title, ticker, sector in discovery[
        [text_col, "ticker", "sector"]
    ].itertuples(index=False, name=None):
        document_terms: set[str] = set()
        for match in TOKEN_PATTERN.finditer(raw_title):
            term = match.group(0).lower()
            document_terms.add(term)
            record = records.setdefault(
                term,
                {
                    "frequency": 0,
                    "document_frequency": 0,
                    "tickers": set(),
                    "sectors": set(),
                    "examples": [],
                    "example_set": set(),
                },
            )
            record["frequency"] = int(record["frequency"]) + 1
            record["tickers"].add(str(ticker))
            record["sectors"].add(str(sector))
            if (
                raw_title not in record["example_set"]
                and len(record["examples"]) < examples_per_term
            ):
                record["examples"].append(raw_title)
                record["example_set"].add(raw_title)
        for term in document_terms:
            records[term]["document_frequency"] = (
                int(records[term]["document_frequency"]) + 1
            )

    window_start = start.date().isoformat()
    window_end = end.date().isoformat()
    n_headlines = len(discovery)
    rows: list[dict[str, object]] = []
    for term, record in records.items():
        in_lexicon = term in lexicon
        is_stopword, review_eligible, exclusion_reason = _review_diagnostics(term)
        rows.append(
            {
                "term": term,
                "frequency": int(record["frequency"]),
                "document_frequency": int(record["document_frequency"]),
                "in_lexicon": in_lexicon,
                "current_valence": lexicon[term] if in_lexicon else math.nan,
                "n_tickers": len(record["tickers"]),
                "n_sectors": len(record["sectors"]),
                "example_headlines": " || ".join(record["examples"]),
                "window_start": window_start,
                "window_end": window_end,
                "n_headlines_in_window": n_headlines,
                "is_stopword": is_stopword,
                "review_eligible": review_eligible,
                "exclusion_reason": exclusion_reason,
            }
        )

    result = pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)
    result = result.sort_values(
        ["frequency", "term"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    if top_n is not None:
        result = result.head(top_n).reset_index(drop=True)
    return result


def read_candidate_terms(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Read a candidate CSV while preserving literal terms such as ``nan``."""
    result = pd.read_csv(path, keep_default_na=False)
    missing = set(CANDIDATE_COLUMNS) - set(result.columns)
    if missing:
        raise ValueError(f"candidate table is missing columns: {sorted(missing)}")
    result = result[CANDIDATE_COLUMNS].copy()
    result["term"] = result["term"].astype(str)
    for column in [
        "frequency",
        "document_frequency",
        "n_tickers",
        "n_sectors",
        "n_headlines_in_window",
    ]:
        result[column] = pd.to_numeric(result[column], errors="raise")
    result["current_valence"] = pd.to_numeric(
        result["current_valence"], errors="coerce"
    )
    for column in ["in_lexicon", "is_stopword", "review_eligible"]:
        values = result[column].astype(str).str.lower()
        if not values.isin({"true", "false"}).all():
            raise ValueError(f"candidate table has invalid boolean values in {column}")
        result[column] = values.eq("true")
    return result


def write_candidate_terms(
    candidates: pd.DataFrame, path: str | os.PathLike[str]
) -> pd.DataFrame:
    """Write candidates and verify a safe CSV round trip."""
    missing = set(CANDIDATE_COLUMNS) - set(candidates.columns)
    if missing:
        raise ValueError(f"candidate table is missing columns: {sorted(missing)}")
    candidates[CANDIDATE_COLUMNS].to_csv(path, index=False)
    restored = read_candidate_terms(path)
    if len(restored) != len(candidates):
        raise RuntimeError("candidate CSV row count changed during round trip")
    if restored["term"].isna().any():
        raise RuntimeError("candidate CSV contains missing terms after round trip")
    expected_nan_terms = int(candidates["term"].astype(str).eq("nan").sum())
    restored_nan_terms = int(restored["term"].eq("nan").sum())
    if restored_nan_terms != expected_nan_terms:
        raise RuntimeError("literal candidate term 'nan' was not preserved")
    return restored


# ============================================================================
# 2021-2023 extended sentiment build: dual scoring, aggregation, standardising
# ============================================================================

DUAL_SCORE_COLUMNS = [
    "baseline_compound",
    "extended_compound",
    "baseline_score_100",
    "extended_score_100",
    "score_delta",
    "custom_term_hit",
    "custom_term_hit_count",
    "matched_custom_terms",
]

SENTIMENT_BANDS = ("extreme_fear", "fear", "neutral", "greed", "extreme_greed")

VADER_NEUTRAL_LOWER = -0.05
VADER_NEUTRAL_UPPER = 0.05


def assert_application_period_only(dates: pd.Series, *, context: str) -> None:
    """Raise if any date in ``dates`` falls outside the frozen 2021-2023 window.

    2020 is discovery-only (candidate-term mining and lexicon design); every
    official sentiment result in this build - indexes, comparison tables, and
    figures - is restricted to APPLICATION_START_DATE..APPLICATION_END_DATE.
    """
    normalized = pd.to_datetime(dates, errors="coerce")
    if getattr(normalized.dt, "tz", None) is not None:
        normalized = normalized.dt.tz_convert(None)
    normalized = normalized.dt.normalize()
    start = pd.Timestamp(APPLICATION_START_DATE)
    end = pd.Timestamp(APPLICATION_END_DATE)
    outside = ~normalized.between(start, end, inclusive="both")
    if outside.any():
        bad_dates = sorted(pd.Series(normalized.loc[outside].dropna().unique()))
        raise AssertionError(
            f"{context} contains {int(outside.sum())} row(s) outside the frozen "
            f"application period {APPLICATION_START_DATE} to {APPLICATION_END_DATE}: "
            f"{[str(pd.Timestamp(d).date()) for d in bad_dates[:5]]}"
        )


def _score_100(compound: pd.Series | float) -> pd.Series | float:
    return (compound + 1.0) / 2.0 * 100.0


def score_headlines_dual(
    panel: pd.DataFrame,
    *,
    text_col: str = "raw_title",
    custom_lexicon: Mapping[str, float] = sentiment_lexicon.CUSTOM_SENTIMENT_LEXICON,
    baseline_analyzer: object | None = None,
    extended_analyzer: object | None = None,
) -> pd.DataFrame:
    """Score every unmodified raw title with baseline and extended FinVADER.

    Baseline: native VADER + SentiBignomics x0.1 + Henry (matches FinVADER's
    public scorer - see the parity test in tests/test_sentiment.py). Extended:
    baseline plus ``custom_lexicon`` (RiskBridge's 30-term lexicon by
    default), merged raw. The only difference between the two models is the
    presence of the custom terms.

    Distinct raw titles are scored once each and merged back onto every row
    that shares that exact title; ``text_col`` is never modified (original
    casing, punctuation, negation, and intensifiers are preserved).
    """
    if text_col not in panel.columns:
        raise ValueError(f"headline panel is missing required column: {text_col!r}")

    titles = panel[text_col]
    invalid = titles.map(lambda value: not isinstance(value, str) or not value.strip())
    if invalid.any():
        raise ValueError(
            f"headline panel contains {int(invalid.sum())} missing, blank, or "
            f"non-string value(s) in {text_col!r}"
        )

    if baseline_analyzer is None:
        baseline_analyzer = build_analyzer(custom_lexicon=None)
    if extended_analyzer is None:
        extended_analyzer = build_analyzer(custom_lexicon=custom_lexicon)

    custom_terms_lower = {str(term).lower() for term in custom_lexicon}

    distinct_titles = titles.drop_duplicates().tolist()
    score_rows: list[dict[str, object]] = []
    for title in distinct_titles:
        baseline_compound = float(baseline_analyzer.polarity_scores(title)["compound"])
        extended_compound = float(extended_analyzer.polarity_scores(title)["compound"])
        matched = sorted(set(vader_lookup_tokens(title)) & custom_terms_lower)
        score_rows.append(
            {
                text_col: title,
                "baseline_compound": baseline_compound,
                "extended_compound": extended_compound,
                "baseline_score_100": _score_100(baseline_compound),
                "extended_score_100": _score_100(extended_compound),
                "score_delta": extended_compound - baseline_compound,
                "custom_term_hit": len(matched) > 0,
                "custom_term_hit_count": len(matched),
                "matched_custom_terms": ", ".join(matched),
            }
        )
    scores = pd.DataFrame(score_rows, columns=[text_col, *DUAL_SCORE_COLUMNS])

    result = panel.merge(scores, on=text_col, how="left", validate="many_to_one")
    if result[DUAL_SCORE_COLUMNS].isna().any().any():
        raise RuntimeError(
            "dual scoring merge produced missing values for some headline rows"
        )
    return result


def ticker_day_sentiment(
    scored_headlines: pd.DataFrame, *, date_col: str = "trading_date"
) -> pd.DataFrame:
    """Aggregate headline-level dual scores to ticker-trading-day means.

    Ticker-days with no headlines simply do not appear here - no fabricated
    neutral (0) value and no carry-forward from a prior day. Callers that need
    an explicit missing row per calendar date should reindex the result
    themselves (see ``sector_day_sentiment``, which does this at the sector
    level).
    """
    required = {date_col, "ticker", "sector", "baseline_compound", "extended_compound"}
    missing = required - set(scored_headlines.columns)
    if missing:
        raise ValueError(f"scored headlines missing required columns: {sorted(missing)}")

    grouped = (
        scored_headlines.groupby([date_col, "ticker", "sector"], as_index=False)
        .agg(
            baseline_compound=("baseline_compound", "mean"),
            extended_compound=("extended_compound", "mean"),
            headline_count=("baseline_compound", "size"),
        )
        .rename(columns={date_col: "date"})
    )
    return grouped.sort_values(["date", "sector", "ticker"]).reset_index(drop=True)


def sector_day_sentiment(
    ticker_day: pd.DataFrame,
    *,
    trading_dates: Iterable[pd.Timestamp],
    sector_universe: Mapping[str, int],
) -> pd.DataFrame:
    """Equal-weight sector-day sentiment across tickers that had news that day.

    Averages only ACTIVE tickers (those with a ticker-day value) so a
    heavily-covered ticker cannot dominate a sector-day average the way
    averaging every headline directly would. Sector-days with zero active
    tickers are explicit NaN rows over the full ``trading_dates`` x
    ``sector_universe`` grid - not dropped, not filled with 0 or a prior
    value - so "no news" stays distinguishable from "neutral news".
    """
    required = {"date", "ticker", "sector", "baseline_compound", "extended_compound", "headline_count"}
    missing = required - set(ticker_day.columns)
    if missing:
        raise ValueError(f"ticker-day panel missing required columns: {sorted(missing)}")

    sectors = sorted(sector_universe.keys())
    dates = pd.DatetimeIndex(
        sorted(pd.to_datetime(pd.Series(list(trading_dates))).unique())
    )
    full_index = pd.MultiIndex.from_product([dates, sectors], names=["date", "sector"])

    active = (
        ticker_day.groupby(["date", "sector"], as_index=False)
        .agg(
            baseline_compound=("baseline_compound", "mean"),
            extended_compound=("extended_compound", "mean"),
            headline_count=("headline_count", "sum"),
            active_ticker_count=("ticker", "nunique"),
        )
        .set_index(["date", "sector"])
    )

    result = active.reindex(full_index).reset_index()
    result["headline_count"] = result["headline_count"].fillna(0).astype(int)
    result["active_ticker_count"] = result["active_ticker_count"].fillna(0).astype(int)
    result["sector_universe_size"] = result["sector"].map(sector_universe).astype(int)
    result["coverage_ratio"] = result["active_ticker_count"] / result["sector_universe_size"]
    result["baseline_score_100"] = _score_100(result["baseline_compound"])
    result["extended_score_100"] = _score_100(result["extended_compound"])

    return result[
        [
            "date",
            "sector",
            "baseline_compound",
            "extended_compound",
            "baseline_score_100",
            "extended_score_100",
            "headline_count",
            "active_ticker_count",
            "sector_universe_size",
            "coverage_ratio",
        ]
    ]


def expanding_standardize(
    df: pd.DataFrame,
    value_col: str,
    *,
    date_col: str = "date",
    group_col: str = "sector",
    min_periods: int = 60,
) -> pd.Series:
    """Prior-history-only expanding z-score, independently per group.

    For each group (sector), sorted by date: ``history = value.shift(1)``,
    then ``z_t = (value_t - expanding_mean(history)) / expanding_std(history, ddof=1)``.
    Day t's z-score therefore uses only day t-1 and earlier values - never the
    current or a future observation. No-news days already carry NaN in
    ``value_col``, and pandas' expanding mean/std skip NaN and do not count it
    toward ``min_periods``, so missing days neither enter the history nor get
    a fabricated value. A group's first ``min_periods`` non-missing prior
    observations produce NaN z-scores (never backfilled from 2020 or any
    other out-of-window source).
    """
    required = {date_col, group_col, value_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"dataframe missing required columns: {sorted(missing)}")

    ordered = df.sort_values([group_col, date_col])

    def _group_z(series: pd.Series) -> pd.Series:
        history = series.shift(1)
        expanding_mean = history.expanding(min_periods=min_periods).mean()
        expanding_sd = history.expanding(min_periods=min_periods).std(ddof=1)
        return (series - expanding_mean) / expanding_sd

    # groupby(...).apply() is ambiguous about Series-vs-DataFrame reshaping
    # (especially with a single group), so use transform(), which is built
    # for exactly this "windowed function per group, same-length result" case.
    z = ordered.groupby(group_col)[value_col].transform(_group_z)
    return z.reindex(df.index)


def classify_sentiment_band(z: pd.Series) -> pd.Series:
    """Five-band classification of an expanding z-score series.

    z < -1.5: extreme_fear; -1.5 <= z < -0.5: fear; -0.5 <= z <= 0.5: neutral;
    0.5 < z <= 1.5: greed; z > 1.5: extreme_greed. NaN stays NaN - a missing or
    insufficient-history period is unclassified, never silently "neutral".
    """
    conditions = [
        z < -1.5,
        (z >= -1.5) & (z < -0.5),
        (z >= -0.5) & (z <= 0.5),
        (z > 0.5) & (z <= 1.5),
        z > 1.5,
    ]
    # np.select rejects mixing string choices with a float NaN default across
    # numpy versions, so build the object-dtype result by hand instead.
    band = pd.Series(np.nan, index=z.index, dtype=object)
    for condition, label in zip(conditions, SENTIMENT_BANDS):
        band.loc[condition] = label
    return band


def add_expanding_zscores(
    sector_day: pd.DataFrame, *, min_periods: int = 60
) -> pd.DataFrame:
    """Add baseline/extended expanding z-scores and five-band labels.

    Completes the required ``sector_sentiment_index.csv`` schema: date,
    sector, baseline/extended compound, baseline/extended score_100,
    baseline/extended_z_expanding, baseline/extended_band, plus the coverage
    columns already produced by ``sector_day_sentiment``.
    """
    result = sector_day.copy()
    result["baseline_z_expanding"] = expanding_standardize(
        result, "baseline_compound", min_periods=min_periods
    )
    result["extended_z_expanding"] = expanding_standardize(
        result, "extended_compound", min_periods=min_periods
    )
    result["baseline_band"] = classify_sentiment_band(result["baseline_z_expanding"])
    result["extended_band"] = classify_sentiment_band(result["extended_z_expanding"])
    return result[
        [
            "date",
            "sector",
            "baseline_compound",
            "extended_compound",
            "baseline_score_100",
            "extended_score_100",
            "baseline_z_expanding",
            "extended_z_expanding",
            "baseline_band",
            "extended_band",
            "headline_count",
            "active_ticker_count",
            "sector_universe_size",
            "coverage_ratio",
        ]
    ]


# ============================================================================
# Baseline-extended automated comparison (no manual headline audit - see
# module docstring in scripts/run_part_b.py for what these tables may and may
# not claim)
# ============================================================================


def custom_lexicon_coverage_table(scored_headlines_2021_2023: pd.DataFrame) -> pd.DataFrame:
    """Tidy long-format coverage: overall, by sector, by year, and the
    per-headline multi-hit distribution. Caller must already have restricted
    the input to the 2021-2023 application period.
    """
    df = scored_headlines_2021_2023
    total = len(df)
    rows: list[dict[str, object]] = []

    def _add(scope: str, group: str, metric: str, value: object) -> None:
        rows.append({"scope": scope, "group": group, "metric": metric, "value": value})

    hit_count = int(df["custom_term_hit"].sum())
    changed = int((df["baseline_compound"] != df["extended_compound"]).sum())
    _add("overall", "", "headline_count", total)
    _add("overall", "", "custom_hit_count", hit_count)
    _add("overall", "", "custom_hit_share", hit_count / total if total else np.nan)
    _add("overall", "", "score_changed_count", changed)
    _add("overall", "", "score_changed_share", changed / total if total else np.nan)

    for sector, g in df.groupby("sector"):
        n = len(g)
        h = int(g["custom_term_hit"].sum())
        _add("sector", str(sector), "headline_count", n)
        _add("sector", str(sector), "custom_hit_count", h)
        _add("sector", str(sector), "custom_hit_share", h / n if n else np.nan)

    year_series = pd.to_datetime(df["trading_date"]).dt.year
    for year, g in df.groupby(year_series):
        n = len(g)
        h = int(g["custom_term_hit"].sum())
        _add("year", str(int(year)), "headline_count", n)
        _add("year", str(int(year)), "custom_hit_count", h)
        _add("year", str(int(year)), "custom_hit_share", h / n if n else np.nan)

    hit_dist = df["custom_term_hit_count"].value_counts().sort_index()
    for n_hits, count in hit_dist.items():
        _add("hit_count_distribution", str(int(n_hits)), "headline_count", int(count))

    return pd.DataFrame(rows, columns=["scope", "group", "metric", "value"])


def custom_term_impact_table(
    scored_headlines_2021_2023: pd.DataFrame,
    *,
    text_col: str = "raw_title",
    custom_lexicon: Mapping[str, float] = sentiment_lexicon.CUSTOM_SENTIMENT_LEXICON,
) -> pd.DataFrame:
    """Per-custom-term frequency, document frequency, and conditional impact.

    ``frequency`` counts raw VADER-lookup-token occurrences (a term repeated
    within one headline counts once per occurrence); ``document_frequency``
    counts headline rows containing the term at least once - the same
    convention as ``candidate_terms()``.
    """
    df = scored_headlines_2021_2023
    total_headlines = len(df)
    distinct_titles = df[text_col].drop_duplicates().tolist()
    token_counts = {title: Counter(vader_lookup_tokens(title)) for title in distinct_titles}
    custom_terms_lower = {str(term).lower() for term in custom_lexicon}

    stats: dict[str, dict[str, object]] = {
        term: {
            "frequency": 0,
            "document_frequency": 0,
            "tickers": set(),
            "sectors": set(),
            "deltas": [],
        }
        for term in custom_terms_lower
    }
    for title, ticker, sector, delta in df[
        [text_col, "ticker", "sector", "score_delta"]
    ].itertuples(index=False, name=None):
        counter = token_counts[title]
        for term in custom_terms_lower:
            count = counter.get(term, 0)
            if count:
                bucket = stats[term]
                bucket["frequency"] = int(bucket["frequency"]) + count
                bucket["document_frequency"] = int(bucket["document_frequency"]) + 1
                bucket["tickers"].add(str(ticker))
                bucket["sectors"].add(str(sector))
                bucket["deltas"].append(float(delta))

    rows = []
    for term, valence in custom_lexicon.items():
        term_lower = str(term).lower()
        bucket = stats[term_lower]
        doc_freq = int(bucket["document_frequency"])
        deltas = bucket["deltas"]
        rows.append(
            {
                "term": term_lower,
                "valence": float(valence),
                "frequency": int(bucket["frequency"]),
                "document_frequency": doc_freq,
                "share_of_headlines": doc_freq / total_headlines if total_headlines else np.nan,
                "n_tickers": len(bucket["tickers"]),
                "n_sectors": len(bucket["sectors"]),
                "mean_score_delta_when_present": float(np.mean(deltas)) if deltas else np.nan,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["frequency", "term"], ascending=[False, True])
        .reset_index(drop=True)
    )


def headline_sentiment_comparison_table(
    scored_headlines_2021_2023: pd.DataFrame,
) -> pd.DataFrame:
    """Conditional baseline-vs-extended impact over three headline scopes:
    all headlines, custom-term-hit headlines, and headlines whose score
    actually changed. Coverage and impact only - no accuracy claim.
    """
    df = scored_headlines_2021_2023
    scopes = {
        "all_headlines": df,
        "custom_term_hit": df.loc[df["custom_term_hit"]],
        "score_changed": df.loc[df["baseline_compound"] != df["extended_compound"]],
    }
    rows = []
    for scope_name, scope_df in scopes.items():
        n = len(scope_df)
        row: dict[str, object] = {"scope": scope_name, "n_headlines": n}
        if n == 0:
            rows.append(row)
            continue
        delta = scope_df["score_delta"]
        positive = int((delta > 0).sum())
        negative = int((delta < 0).sum())
        unchanged = int((delta == 0).sum())
        nonzero_both = (scope_df["baseline_compound"] != 0) & (scope_df["extended_compound"] != 0)
        sign_differs = np.sign(scope_df["baseline_compound"]) != np.sign(scope_df["extended_compound"])
        sign_flip = int((sign_differs & nonzero_both).sum())
        pearson = scope_df["baseline_compound"].corr(scope_df["extended_compound"], method="pearson")
        spearman = scope_df["baseline_compound"].corr(scope_df["extended_compound"], method="spearman")
        row.update(
            {
                "mean_delta": float(delta.mean()),
                "median_delta": float(delta.median()),
                "mean_abs_delta": float(delta.abs().mean()),
                "delta_p10": float(delta.quantile(0.10)),
                "delta_p25": float(delta.quantile(0.25)),
                "delta_p50": float(delta.quantile(0.50)),
                "delta_p75": float(delta.quantile(0.75)),
                "delta_p90": float(delta.quantile(0.90)),
                "positive_share": positive / n,
                "negative_share": negative / n,
                "no_change_share": unchanged / n,
                "sign_flip_rate": sign_flip / n,
                "pearson_corr": float(pearson) if pd.notna(pearson) else np.nan,
                "spearman_corr": float(spearman) if pd.notna(spearman) else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def largest_sentiment_score_changes_table(
    scored_headlines_2021_2023: pd.DataFrame,
    *,
    top_n: int = 50,
    text_col: str = "raw_title",
) -> pd.DataFrame:
    """The ``top_n`` headlines with the largest |extended - baseline| delta."""
    df = scored_headlines_2021_2023.copy()
    df["abs_score_delta"] = df["score_delta"].abs()
    cols = [
        "trading_date",
        "ticker",
        "sector",
        text_col,
        "baseline_compound",
        "extended_compound",
        "score_delta",
        "matched_custom_terms",
    ]
    return (
        df.sort_values("abs_score_delta", ascending=False)
        .head(top_n)[cols]
        .reset_index(drop=True)
    )


def _vader_band(compound: pd.Series) -> pd.Series:
    """3-way VADER band: negative <= -0.05; neutral in (-0.05, 0.05); positive >= 0.05."""
    conditions = [
        compound <= VADER_NEUTRAL_LOWER,
        (compound > VADER_NEUTRAL_LOWER) & (compound < VADER_NEUTRAL_UPPER),
        compound >= VADER_NEUTRAL_UPPER,
    ]
    band = pd.Series(np.nan, index=compound.index, dtype=object)
    for condition, label in zip(conditions, ["negative", "neutral", "positive"]):
        band.loc[condition] = label
    return band


def neutral_reclassification_summary_table(
    scored_headlines_2021_2023: pd.DataFrame,
) -> pd.DataFrame:
    """Baseline-neutral reclassification among custom-term-hit headlines only.

    Reports two rates - exact-zero reclassification and neutral-VADER-band
    exit - plus the full baseline-to-extended band transition breakdown. These
    are potentially coverage-related neutral cases, not a confirmed
    false-neutral correction: no manual headline audit was performed.
    """
    df = scored_headlines_2021_2023
    hit = df.loc[df["custom_term_hit"]]

    exact_zero_denominator = int((hit["baseline_compound"] == 0.0).sum())
    exact_zero_numerator = int(
        ((hit["baseline_compound"] == 0.0) & (hit["extended_compound"] != 0.0)).sum()
    )
    exact_zero_rate = (
        exact_zero_numerator / exact_zero_denominator if exact_zero_denominator else np.nan
    )

    baseline_band = _vader_band(hit["baseline_compound"])
    extended_band = _vader_band(hit["extended_compound"])

    neutral_mask = baseline_band == "neutral"
    neutral_band_denominator = int(neutral_mask.sum())
    neutral_band_numerator = int((neutral_mask & (extended_band != "neutral")).sum())
    neutral_band_exit_rate = (
        neutral_band_numerator / neutral_band_denominator if neutral_band_denominator else np.nan
    )

    n_neutral_to_positive = int(((baseline_band == "neutral") & (extended_band == "positive")).sum())
    n_neutral_to_negative = int(((baseline_band == "neutral") & (extended_band == "negative")).sum())
    n_pos_or_neg_to_neutral = int(
        (baseline_band.isin(["positive", "negative"]) & (extended_band == "neutral")).sum()
    )
    n_sign_flip = int(
        (
            ((baseline_band == "positive") & (extended_band == "negative"))
            | ((baseline_band == "negative") & (extended_band == "positive"))
        ).sum()
    )

    rows = [
        {"metric": "custom_term_hit_headlines", "value": len(hit)},
        {"metric": "exact_zero_denominator", "value": exact_zero_denominator},
        {"metric": "exact_zero_numerator", "value": exact_zero_numerator},
        {"metric": "exact_zero_reclassification_rate", "value": exact_zero_rate},
        {"metric": "neutral_band_denominator", "value": neutral_band_denominator},
        {"metric": "neutral_band_numerator", "value": neutral_band_numerator},
        {"metric": "neutral_band_exit_rate", "value": neutral_band_exit_rate},
        {"metric": "transition_neutral_to_positive", "value": n_neutral_to_positive},
        {"metric": "transition_neutral_to_negative", "value": n_neutral_to_negative},
        {"metric": "transition_pos_or_neg_to_neutral", "value": n_pos_or_neg_to_neutral},
        {"metric": "transition_sign_flip_positive_negative", "value": n_sign_flip},
    ]
    return pd.DataFrame(rows)


def daily_overall_sentiment_aggregate(sector_index: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight, across-sector daily aggregate: one row per date.

    Each date's aggregate value averages only the sectors with a valid
    (non-missing) reading that day for that specific column - a sector with
    no news, or still inside its expanding warm-up, is excluded from that
    day's average rather than contributing a fabricated value (this is
    exactly what pandas' default skip-NaN ``mean()`` already does per
    column). Raw compound and standardised z columns are aggregated
    independently, because a sector can have a valid raw compound before it
    has accumulated 60 prior observations for a z-score.

    The z aggregate is NOT re-standardised - it stays the equal-weight mean
    of each sector's own already-computed ``*_z_expanding``. The existing
    five-band thresholds (``classify_sentiment_band``) are applied to that
    aggregate z-series so the "Overall" scope gets bands on the same basis as
    every individual sector.

    This is the single shared source for both the "Overall" row of
    ``sector_sentiment_model_comparison_table`` and Figure 7's Overall
    aggregate panel in scripts/run_part_b.py - they must never diverge.
    """
    required = {
        "date",
        "baseline_compound",
        "extended_compound",
        "baseline_z_expanding",
        "extended_z_expanding",
    }
    missing = required - set(sector_index.columns)
    if missing:
        raise ValueError(f"sector_index missing required columns: {sorted(missing)}")

    raw = sector_index.groupby("date", as_index=False)[
        ["baseline_compound", "extended_compound"]
    ].mean()
    z = sector_index.groupby("date", as_index=False)[
        ["baseline_z_expanding", "extended_z_expanding"]
    ].mean()
    aggregate = raw.merge(z, on="date", how="outer").sort_values("date").reset_index(drop=True)
    aggregate["baseline_band"] = classify_sentiment_band(aggregate["baseline_z_expanding"])
    aggregate["extended_band"] = classify_sentiment_band(aggregate["extended_z_expanding"])
    return aggregate


def sector_sentiment_model_comparison_table(
    sector_index_2021_2023: pd.DataFrame,
    scored_headlines_2021_2023: pd.DataFrame,
) -> pd.DataFrame:
    """Compare baseline vs extended by sector and overall - RAW index first,
    then the expanding z-score, never only the standardised result.

    The "Overall" row's mean/difference/correlation/band-change/extreme-date
    metrics are computed from ``daily_overall_sentiment_aggregate`` (one
    equal-sector-weight observation per date), not from pooling every
    (date, sector) row together - pooling would let sectors with more
    news-covered days dominate and would not match a daily aggregate time
    series such as Figure 7's Overall panel. Per-sector rows are unaffected:
    each still uses its own (date, sector) rows directly, as before.
    """
    df = sector_index_2021_2023
    daily_overall = daily_overall_sentiment_aggregate(df)
    hit_rate_by_sector = scored_headlines_2021_2023.groupby("sector")["custom_term_hit"].mean()
    overall_hit_rate = float(scored_headlines_2021_2023["custom_term_hit"].mean())

    def _extreme_dates(band: pd.Series, date: pd.Series) -> set:
        mask = band.isin(["extreme_fear", "extreme_greed"])
        return set(pd.to_datetime(date.loc[mask]).dt.normalize())

    def _jaccard(a: set, b: set) -> float:
        union = a | b
        return len(a & b) / len(union) if union else np.nan

    def _row(scope_name: str, scope_df: pd.DataFrame, stats_df: pd.DataFrame) -> dict[str, object]:
        """``scope_df`` supplies headline volume/coverage/hit-rate context;
        ``stats_df`` supplies the mean/difference/correlation/band series
        (for "Overall" this is the daily equal-weight aggregate; for a single
        sector the two are the same (date, sector) rows).
        """
        raw = stats_df.dropna(subset=["baseline_compound", "extended_compound"])
        z = stats_df.dropna(subset=["baseline_z_expanding", "extended_z_expanding"])
        band_common = stats_df.dropna(subset=["baseline_band", "extended_band"])

        raw_pearson = raw["baseline_compound"].corr(raw["extended_compound"], method="pearson") if len(raw) else np.nan
        raw_spearman = raw["baseline_compound"].corr(raw["extended_compound"], method="spearman") if len(raw) else np.nan
        z_pearson = z["baseline_z_expanding"].corr(z["extended_z_expanding"], method="pearson") if len(z) else np.nan
        z_spearman = z["baseline_z_expanding"].corr(z["extended_z_expanding"], method="spearman") if len(z) else np.nan
        band_change_rate = (
            float((band_common["baseline_band"] != band_common["extended_band"]).mean())
            if len(band_common) else np.nan
        )
        baseline_extreme = _extreme_dates(stats_df["baseline_band"], stats_df["date"])
        extended_extreme = _extreme_dates(stats_df["extended_band"], stats_df["date"])
        hit_rate = overall_hit_rate if scope_name == "Overall" else float(
            hit_rate_by_sector.get(scope_name, np.nan)
        )

        return {
            "scope": scope_name,
            "custom_term_hit_rate": hit_rate,
            "headline_count": int(scope_df["headline_count"].sum()),
            "avg_active_ticker_coverage": float(scope_df["coverage_ratio"].mean()),
            "raw_baseline_mean": float(raw["baseline_compound"].mean()) if len(raw) else np.nan,
            "raw_extended_mean": float(raw["extended_compound"].mean()) if len(raw) else np.nan,
            "raw_mean_difference": float((raw["extended_compound"] - raw["baseline_compound"]).mean()) if len(raw) else np.nan,
            "raw_mean_abs_difference": float((raw["extended_compound"] - raw["baseline_compound"]).abs().mean()) if len(raw) else np.nan,
            "raw_pearson_corr": float(raw_pearson) if pd.notna(raw_pearson) else np.nan,
            "raw_spearman_corr": float(raw_spearman) if pd.notna(raw_spearman) else np.nan,
            "z_baseline_mean": float(z["baseline_z_expanding"].mean()) if len(z) else np.nan,
            "z_extended_mean": float(z["extended_z_expanding"].mean()) if len(z) else np.nan,
            "z_mean_difference": float((z["extended_z_expanding"] - z["baseline_z_expanding"]).mean()) if len(z) else np.nan,
            "z_mean_abs_difference": float((z["extended_z_expanding"] - z["baseline_z_expanding"]).abs().mean()) if len(z) else np.nan,
            "z_pearson_corr": float(z_pearson) if pd.notna(z_pearson) else np.nan,
            "z_spearman_corr": float(z_spearman) if pd.notna(z_spearman) else np.nan,
            "z_band_change_rate": band_change_rate,
            "z_extreme_date_overlap_jaccard": _jaccard(baseline_extreme, extended_extreme),
        }

    rows = [_row("Overall", df, daily_overall)]
    for sector, g in df.groupby("sector"):
        rows.append(_row(str(sector), g, g))
    return pd.DataFrame(rows)

