"""Offline tests for the FinVADER baseline and 2020 candidate discovery."""
from __future__ import annotations

import builtins
import pathlib
import subprocess
import sys
from types import SimpleNamespace

import matplotlib

# Must run before `from scripts import run_part_b` (which imports
# matplotlib.pyplot) and before any other matplotlib import in this file.
# Without this, the default GUI backend (e.g. macosx) can load
# matplotlib.backends._macosx in a headless test run and crash the interpreter
# with "Fatal Python error: Aborted" instead of raising a catchable exception.
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import run_part_b  # noqa: E402
from src import features, sentiment, sentiment_lexicon  # noqa: E402


def _headline_panel(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _candidate_row(**overrides: object) -> pd.DataFrame:
    row = {
        "term": "outlook",
        "frequency": 1,
        "document_frequency": 1,
        "in_lexicon": False,
        "current_valence": float("nan"),
        "n_tickers": 1,
        "n_sectors": 1,
        "example_headlines": "The outlook is strong!",
        "window_start": "2020-01-01",
        "window_end": "2020-12-31",
        "n_headlines_in_window": 1,
        "is_stopword": False,
        "review_eligible": True,
        "exclusion_reason": "",
    }
    row.update(overrides)
    return pd.DataFrame([row], columns=sentiment.CANDIDATE_COLUMNS)


def test_guarded_import_restores_download_after_success(monkeypatch):
    import nltk

    original_download = nltk.download
    blocked_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    fake_module = SimpleNamespace(finvader=lambda *args, **kwargs: 0.0)

    monkeypatch.setattr(nltk.data, "find", lambda path: pathlib.Path("local"))

    def fake_import(name: str) -> object:
        assert name == "finvader"
        blocked_calls.append((("vader_lexicon",), {}))
        assert nltk.download("vader_lexicon") is True
        return fake_module

    monkeypatch.setattr(sentiment.importlib, "import_module", fake_import)
    assert sentiment.load_finvader_module() is fake_module
    assert nltk.download is original_download
    assert blocked_calls == [(('vader_lexicon',), {})]


def test_guarded_import_restores_download_after_import_failure(monkeypatch):
    import nltk

    original_download = nltk.download
    monkeypatch.setattr(nltk.data, "find", lambda path: pathlib.Path("local"))

    def failed_import(name: str) -> object:
        assert nltk.download is not original_download
        raise ImportError("synthetic import failure")

    monkeypatch.setattr(sentiment.importlib, "import_module", failed_import)
    with pytest.raises(ImportError, match="synthetic import failure"):
        sentiment.load_finvader_module()
    assert nltk.download is original_download


def test_guarded_import_missing_resource_never_downloads(monkeypatch):
    import nltk

    download_calls: list[object] = []
    import_calls: list[str] = []

    def missing_resource(path: str) -> None:
        raise LookupError(path)

    def forbidden_download(*args: object, **kwargs: object) -> bool:
        download_calls.append((args, kwargs))
        return False

    def forbidden_import(name: str) -> object:
        import_calls.append(name)
        raise AssertionError("finvader import must not be attempted")

    monkeypatch.setattr(nltk.data, "find", missing_resource)
    monkeypatch.setattr(nltk, "download", forbidden_download)
    monkeypatch.setattr(sentiment.importlib, "import_module", forbidden_import)
    with pytest.raises(LookupError, match="not installed locally"):
        sentiment.load_finvader_module()
    assert download_calls == []
    assert import_calls == []


def test_load_finvader_scorer_returns_public_function(monkeypatch):
    def public_finvader(*args: object, **kwargs: object) -> float:
        return 0.0

    monkeypatch.setattr(
        sentiment,
        "load_finvader_module",
        lambda: SimpleNamespace(finvader=public_finvader),
    )
    assert sentiment.load_finvader_scorer() is public_finvader


def test_scorer_receives_raw_title_exactly_and_never_calls_top_terms(monkeypatch):
    raw_title = "Company is not VERY profitable!!!"
    panel = pd.DataFrame(
        {"raw_title": [raw_title], "discovery_text": ["company profitable"]}
    )
    received: list[tuple[str, dict[str, object]]] = []

    def fake_scorer(text: str, **kwargs: object) -> float:
        received.append((text, kwargs))
        return -0.25

    monkeypatch.setattr(
        features,
        "top_terms",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("features.top_terms must not be called")
        ),
    )
    result = sentiment.score_headlines(panel, scorer=fake_scorer)

    assert received == [
        (
            raw_title,
            {
                "indicator": "compound",
                "use_sentibignomics": True,
                "use_henry": True,
            },
        )
    ]
    assert received[0][0] == raw_title
    assert "not" in received[0][0]
    assert "VERY" in received[0][0]
    assert "!!!" in received[0][0]
    assert result.loc[0, "raw_title"] == raw_title


@pytest.mark.parametrize("invalid_title", ["", "   ", None, 123])
def test_scorer_rejects_invalid_raw_titles(invalid_title):
    with pytest.raises(ValueError, match="missing, blank, or non-string"):
        sentiment.score_headlines(
            pd.DataFrame({"raw_title": [invalid_title]}), scorer=lambda *args, **kwargs: 0.0
        )


@pytest.mark.parametrize(
    "invalid_score", [None, "0.1", True, float("nan"), float("inf"), float("-inf")]
)
def test_scorer_rejects_invalid_scores(invalid_score):
    with pytest.raises(ValueError, match="non-finite or non-numeric"):
        sentiment.score_headlines(
            pd.DataFrame({"raw_title": ["Valid title"]}),
            scorer=lambda *args, **kwargs: invalid_score,
        )


def test_scorer_is_deterministic_for_same_input():
    def deterministic_scorer(text: str, **kwargs: object) -> float:
        return 0.375 if text == "Same raw title!" else 0.0

    panel = pd.DataFrame({"raw_title": ["Same raw title!", "Same raw title!"]})
    result = sentiment.score_headlines(panel, scorer=deterministic_scorer)
    assert result["sentiment_score"].tolist() == [0.375, 0.375]


def test_effective_lexicon_uses_finvader_update_order():
    module = SimpleNamespace(
        lexicon1=lambda: {"senti_override": 20.0, "shared": 30.0},
        lexicon2=lambda: {"shared": -1.5, "henry_only": 1.5},
    )
    analyzer = SimpleNamespace(
        lexicon={"base_only": 0.5, "senti_override": 0.4, "shared": 0.2}
    )

    result = sentiment.effective_finvader_lexicon(
        module=module, analyzer_factory=lambda: analyzer
    )

    assert result["base_only"] == pytest.approx(0.5)
    assert result["senti_override"] == pytest.approx(2.0)
    assert result["shared"] == pytest.approx(-1.5)
    assert result["henry_only"] == pytest.approx(1.5)


def test_candidate_document_frequency_counts_rows_not_unique_title_strings(
    monkeypatch,
):
    title = "Profit profit outlook"
    panel = _headline_panel(
        [
            {
                "date": "2020-01-02",
                "trading_date": "2020-01-02",
                "ticker": "AAA",
                "sector": "Tech",
                "raw_title": title,
            },
            {
                "date": "2020-01-03",
                "trading_date": "2020-01-03",
                "ticker": "BBB",
                "sector": "Health",
                "raw_title": title,
            },
        ]
    )
    original = panel.copy(deep=True)
    monkeypatch.setattr(
        features,
        "top_terms",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("features.top_terms must not be called")
        ),
    )

    result = sentiment.candidate_terms(panel, baseline_lexicon={})
    profit = result.set_index("term").loc["profit"]

    assert profit["frequency"] == 4
    assert profit["document_frequency"] == 2
    assert profit["n_headlines_in_window"] == 2
    assert profit["n_tickers"] == 2
    assert profit["n_sectors"] == 2
    assert profit["example_headlines"] == title
    pdt.assert_frame_equal(panel, original)


def test_candidate_schema_valence_sort_and_closed_window():
    panel = _headline_panel(
        [
            {"date": "2019-12-31", "ticker": "OLD", "sector": "Old", "raw_title": "outside"},
            {
                "date": "2020-01-01",
                "ticker": "AAA",
                "sector": "Tech",
                "raw_title": "Known mystery mystery",
            },
            {"date": "2020-12-31", "ticker": "BBB", "sector": "Health", "raw_title": "Known alpha"},
            {
                "date": "2021-01-01",
                "ticker": "HOLD",
                "sector": "Holdout",
                "raw_title": "mystery holdoutword",
            },
        ]
    )
    result = sentiment.candidate_terms(panel, baseline_lexicon={"known": 1.25})

    assert result.columns.tolist() == sentiment.CANDIDATE_COLUMNS
    assert result["term"].tolist()[:3] == ["known", "mystery", "alpha"]
    indexed = result.set_index("term")
    assert indexed.loc["known", "in_lexicon"]
    assert indexed.loc["known", "current_valence"] == pytest.approx(1.25)
    assert not indexed.loc["mystery", "in_lexicon"]
    assert pd.isna(indexed.loc["mystery", "current_valence"])
    assert indexed.loc["known", "window_start"] == "2020-01-01"
    assert indexed.loc["known", "window_end"] == "2020-12-31"
    assert indexed.loc["known", "n_headlines_in_window"] == 2
    assert "holdoutword" not in indexed.index


def test_candidate_examples_are_unique_limited_and_deterministic():
    panel = pd.DataFrame(
        [
            {"date": "2020-01-04", "ticker": "D", "sector": "S", "raw_title": "term fourth"},
            {"date": "2020-01-02", "ticker": "B", "sector": "S", "raw_title": "term second"},
            {"date": "2020-01-01", "ticker": "A", "sector": "S", "raw_title": "term first"},
            {"date": "2020-01-03", "ticker": "C", "sector": "S", "raw_title": "term third"},
            {"date": "2020-01-05", "ticker": "E", "sector": "S", "raw_title": "term first"},
        ]
    )
    forward = sentiment.candidate_terms(panel, baseline_lexicon={})
    reversed_result = sentiment.candidate_terms(
        panel.iloc[::-1].reset_index(drop=True), baseline_lexicon={}
    )

    expected = "term first || term second || term third"
    assert forward.set_index("term").loc["term", "example_headlines"] == expected
    assert (
        reversed_result.set_index("term").loc["term", "example_headlines"]
        == expected
    )


def test_candidate_period_uses_original_date_not_trading_date():
    panel = _headline_panel(
        [
            {
                "date": "2020-12-31",
                "trading_date": "2021-01-04",
                "ticker": "AAA",
                "sector": "Tech",
                "raw_title": "includedterm",
            },
            {
                "date": "2021-01-01",
                "trading_date": "2020-12-31",
                "ticker": "HOLD",
                "sector": "Holdout",
                "raw_title": "excludedterm",
            },
        ]
    )
    result = sentiment.candidate_terms(panel, date_col="date", baseline_lexicon={})
    assert result["term"].tolist() == ["includedterm"]


def test_candidate_review_diagnostics_keep_stopwords_and_surface_forms():
    panel = _headline_panel(
        [
            {
                "date": "2020-02-01",
                "ticker": "AAA",
                "sector": "Tech",
                "raw_title": "The outlook covid- sciences'",
            }
        ]
    )
    result = sentiment.candidate_terms(panel, baseline_lexicon={}).set_index("term")

    assert "the" in result.index
    assert result.loc["the", "is_stopword"]
    assert not result.loc["the", "review_eligible"]
    assert result.loc["the", "exclusion_reason"] == "stopword"
    assert result.loc["outlook", "review_eligible"]
    for term in ["covid-", "sciences'"]:
        assert not result.loc[term, "review_eligible"]
        assert "trailing hyphen or apostrophe" in result.loc[term, "exclusion_reason"]
        assert "VADER token mismatch" in result.loc[term, "exclusion_reason"]


def test_candidate_and_vader_token_compatibility_without_resources():
    from nltk.sentiment.vader import SentimentIntensityAnalyzer, VaderConstants

    assert sentiment.token_compatibility_diagnostic("outlook") == {
        "surface_text": "outlook",
        "candidate_tokens": ("outlook",),
        "vader_lookup_tokens": ("outlook",),
        "exact_match": True,
    }
    assert sentiment.token_compatibility_diagnostic("covid-")["exact_match"] is False
    assert sentiment.token_compatibility_diagnostic("sciences'")["exact_match"] is False
    covid_19 = sentiment.token_compatibility_diagnostic("COVID-19")
    assert covid_19["candidate_tokens"] == ("covid-",)
    assert covid_19["vader_lookup_tokens"] == ("covid-19",)
    assert covid_19["exact_match"] is False

    analyzer = SentimentIntensityAnalyzer.__new__(SentimentIntensityAnalyzer)
    analyzer.constants = VaderConstants()
    analyzer.lexicon = {"outlook": 2.0}
    assert analyzer.polarity_scores("outlook")["compound"] > 0


def test_candidate_empty_schema_and_validation():
    panel = pd.DataFrame(columns=["date", "ticker", "sector", "raw_title"])
    empty = sentiment.candidate_terms(panel, baseline_lexicon={})
    assert empty.empty
    assert empty.columns.tolist() == sentiment.CANDIDATE_COLUMNS

    filtered = sentiment.candidate_terms(
        pd.DataFrame(
            [{"date": "2021-01-01", "ticker": "A", "sector": "S", "raw_title": "outside"}]
        ),
        baseline_lexicon={},
    )
    assert filtered.empty
    assert filtered.columns.tolist() == sentiment.CANDIDATE_COLUMNS

    for invalid_top_n in [0, -1, 1.5, True]:
        with pytest.raises(ValueError, match="top_n"):
            sentiment.candidate_terms(
                panel, baseline_lexicon={}, top_n=invalid_top_n
            )
    with pytest.raises(ValueError, match="start_date must be on or before end_date"):
        sentiment.candidate_terms(
            panel,
            baseline_lexicon={},
            start_date="2020-12-31",
            end_date="2020-01-01",
        )


def test_candidate_csv_round_trip_preserves_literal_nan(tmp_path):
    panel = pd.DataFrame(
        [
            {
                "date": "2020-03-01",
                "ticker": "AAA",
                "sector": "Tech",
                "raw_title": "nan outlook",
            }
        ]
    )
    candidates = sentiment.candidate_terms(panel, baseline_lexicon={})
    path = tmp_path / "candidates.csv"
    restored = sentiment.write_candidate_terms(candidates, path)

    assert len(restored) == len(candidates)
    assert restored["term"].isna().sum() == 0
    assert restored["term"].eq("nan").sum() == 1
    assert pd.isna(restored.loc[restored["term"].eq("nan"), "current_valence"]).all()


def test_candidate_only_mode_uses_one_2020_call_and_no_other_build(monkeypatch, tmp_path, capsys):
    equities = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02"]),
            "ticker": ["AAA"],
            "sector": ["Tech"],
            "adjClose": [100.0],
        }
    )
    headlines = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2021-01-01"]),
            "ticker": ["AAA", "HOLD"],
            "sector": ["Tech", "Holdout"],
            "title": ["The outlook is strong!", "holdoutword"],
            "text_raw": ["The outlook is strong!", "holdoutword"],
        }
    )
    aligned = headlines.assign(
        trading_date=pd.to_datetime(["2020-01-02", "2021-01-04"])
    )
    candidate_calls: list[tuple[pd.DataFrame, dict[str, object]]] = []

    monkeypatch.setattr(run_part_b, "TABLES", tmp_path)
    monkeypatch.setattr(run_part_b, "CANDIDATE_PATH", tmp_path / "candidates.csv")
    monkeypatch.setattr(run_part_b.data_access, "load_equity_prices", lambda: equities)
    monkeypatch.setattr(run_part_b.data_access, "load_news_headlines", lambda: headlines)
    monkeypatch.setattr(
        run_part_b.data_access,
        "load_crypto_prices",
        lambda: (_ for _ in ()).throw(AssertionError("crypto build must not run")),
    )
    monkeypatch.setattr(run_part_b.etl, "clean_price_panel", lambda *args, **kwargs: (equities, []))
    monkeypatch.setattr(run_part_b.etl, "clean_headlines", lambda *args, **kwargs: (headlines, []))
    monkeypatch.setattr(
        run_part_b.features,
        "assemble_headline_panel",
        lambda *args, **kwargs: (aligned, pd.DataFrame()),
    )

    def fake_candidates(panel: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
        candidate_calls.append((panel.copy(), kwargs))
        return _candidate_row()

    monkeypatch.setattr(run_part_b.sentiment, "candidate_terms", fake_candidates)
    monkeypatch.setattr(
        run_part_b.sentiment,
        "finvader_metadata",
        lambda: {
            "finvader_version": "1.0.2",
            "indicator": "compound",
            "use_sentibignomics": True,
            "use_henry": True,
        },
    )
    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object):
        if name in {"src.portfolios", "src.fusion", "portfolios", "fusion"}:
            raise AssertionError(f"candidate mode imported {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    run_part_b.main(["--sentiment-candidates-only"])

    assert len(candidate_calls) == 1
    source, kwargs = candidate_calls[0]
    assert source["date"].dt.year.tolist() == [2020]
    assert source["raw_title"].tolist() == ["The outlook is strong!"]
    assert kwargs == {
        "text_col": "raw_title",
        "date_col": "date",
        "start_date": "2020-01-01",
        "end_date": "2020-12-31",
        "top_n": None,
    }
    output = capsys.readouterr().out
    assert "candidate source rows outside discovery window: 0" in output
    assert "top 30 eligible uncovered candidate terms:" in output
    restored = sentiment.read_candidate_terms(tmp_path / "candidates.csv")
    assert restored["term"].tolist() == ["outlook"]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["candidates.csv"]


def test_no_argument_command_keeps_original_dispatch(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(run_part_b, "run_full_build", lambda: calls.append("full"))
    monkeypatch.setattr(
        run_part_b,
        "run_sentiment_candidate_discovery",
        lambda: (_ for _ in ()).throw(AssertionError("candidate mode must not run")),
    )
    run_part_b.main([])
    assert calls == ["full"]


# ============================================================================
# Package import regression: src/sentiment.py must import sentiment_lexicon
# via the package (relative import), not rely on a caller-supplied sys.path
# hack. Both checks spawn a clean subprocess from the project root so this
# test file's own sys.path.insert() calls above cannot mask a regression.
# ============================================================================


def test_package_import_works_from_project_root_without_syspath_hacks():
    result = subprocess.run(
        [sys.executable, "-c", "from src import sentiment; from src import sentiment_lexicon"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_run_part_b_style_bootstrap_still_imports_and_runs_sentiment():
    script = (
        "import sys, pathlib\n"
        "ROOT = pathlib.Path('.').resolve()\n"
        "sys.path.insert(0, str(ROOT))\n"
        "sys.path.insert(0, str(ROOT / 'src'))\n"
        "from src import data_access, etl, features, portfolios, sentiment\n"
        "assert sentiment.build_analyzer(custom_lexicon=None) is not None\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


# ============================================================================
# 2021-2023 extended sentiment build: the 15 required tests
# ============================================================================

REQUIRED_SECTOR_INDEX_COLUMNS = [
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

_PARITY_HEADLINES = [
    "Nvidia shares soar after strong earnings beat",
    "Boeing stock crashes on safety concerns",
    "Fed signals rate cuts, markets rally",
    "Company reports mixed quarterly results",
    "Oil prices plunge amid demand fears",
    "Tech giant announces massive layoffs",
    "Analysts upgrade outlook for retail sector",
    "Bank downgraded after weak loan growth",
]


# 1. 30 terms, unique, native-VADER valence range.
def test_custom_lexicon_has_30_unique_terms_within_vader_valence_range():
    lexicon = sentiment_lexicon.CUSTOM_SENTIMENT_LEXICON
    assert len(lexicon) == 30
    assert len(set(lexicon.keys())) == 30
    for term, value in lexicon.items():
        assert -4.0 <= value <= 4.0, f"{term} valence {value} outside native VADER -4..4"


# 2 & 3 & 4: custom terms merge raw (never x0.1); SentiBignomics still x0.1;
# Henry stays raw and unchanged.
def test_effective_lexicon_merges_custom_terms_raw_sentibignomics_still_scaled_henry_unchanged():
    module = SimpleNamespace(
        lexicon1=lambda: {"senti_term": 10.0},
        lexicon2=lambda: {"henry_term": -1.5},
    )
    analyzer = SimpleNamespace(lexicon={"base_only": 0.5})
    custom = {"crashes": -3.9, "better-than-expected": 2.9}

    result = sentiment.effective_finvader_lexicon(
        module=module, analyzer_factory=lambda: analyzer, custom_lexicon=custom
    )

    assert result["senti_term"] == pytest.approx(1.0)  # SentiBignomics: 10.0 * 0.1
    assert result["henry_term"] == pytest.approx(-1.5)  # Henry: unchanged
    assert result["crashes"] == pytest.approx(-3.9)  # custom raw, NOT -0.39
    assert result["better-than-expected"] == pytest.approx(2.9)  # NOT 0.29


def test_effective_lexicon_raises_on_custom_overlap_instead_of_silently_overwriting():
    module = SimpleNamespace(lexicon1=lambda: {"shared_term": 10.0}, lexicon2=lambda: {})
    analyzer = SimpleNamespace(lexicon={})
    with pytest.raises(ValueError, match="shared_term"):
        sentiment.effective_finvader_lexicon(
            module=module,
            analyzer_factory=lambda: analyzer,
            custom_lexicon={"shared_term": 2.0},
        )


def test_frozen_custom_lexicon_has_zero_overlap_with_real_baseline_effective_lexicon():
    baseline = sentiment.effective_finvader_lexicon()
    overlap = set(sentiment_lexicon.CUSTOM_SENTIMENT_LEXICON) & set(baseline)
    assert overlap == set()


# 5. Baseline analyzer matches the public FinVADER scorer (strict tolerance).
def test_baseline_analyzer_matches_public_finvader_scorer():
    from finvader import finvader

    analyzer = sentiment.build_analyzer(custom_lexicon=None)
    for title in _PARITY_HEADLINES:
        ours = analyzer.polarity_scores(title)["compound"]
        theirs = finvader(title, indicator="compound", use_sentibignomics=True, use_henry=True)
        assert ours == pytest.approx(theirs, abs=1e-9), title


# 6. Headlines with no custom-term hit score identically under both models.
def test_headlines_without_custom_term_hits_score_identically():
    panel = pd.DataFrame(
        {
            "raw_title": [
                "Company reports mixed quarterly results",
                "Regulators review merger filing",
            ],
            "ticker": ["XOM", "GE"],
            "sector": ["Energy", "Industrials"],
            "trading_date": pd.to_datetime(["2021-01-04", "2021-01-05"]),
        }
    )
    scored = sentiment.score_headlines_dual(panel)
    assert not scored["custom_term_hit"].any()
    assert (scored["baseline_compound"] == scored["extended_compound"]).all()
    assert (scored["score_delta"] == 0.0).all()


def test_score_headlines_dual_preserves_raw_text_and_flags_custom_hits():
    panel = pd.DataFrame(
        {
            "raw_title": ["Boeing stock crashes on safety concerns"],
            "ticker": ["BA"],
            "sector": ["Industrials"],
            "trading_date": pd.to_datetime(["2021-01-05"]),
        }
    )
    scored = sentiment.score_headlines_dual(panel)
    assert scored.loc[0, "raw_title"] == "Boeing stock crashes on safety concerns"
    assert scored.loc[0, "custom_term_hit"]
    assert scored.loc[0, "custom_term_hit_count"] == 1
    assert scored.loc[0, "matched_custom_terms"] == "crashes"
    assert scored.loc[0, "extended_compound"] != scored.loc[0, "baseline_compound"]
    for col in ("baseline_score_100", "extended_score_100"):
        expected = (scored.loc[0, col.replace("_score_100", "_compound")] + 1.0) / 2.0 * 100.0
        assert scored.loc[0, col] == pytest.approx(expected)


# 7. Ticker-day sentiment averages headlines within the same ticker-day.
def test_ticker_day_sentiment_averages_headlines_within_ticker_day():
    scored = pd.DataFrame(
        {
            "trading_date": pd.to_datetime(["2021-01-04", "2021-01-04", "2021-01-05"]),
            "ticker": ["NVDA", "NVDA", "NVDA"],
            "sector": ["Tech", "Tech", "Tech"],
            "baseline_compound": [0.5, -0.1, 0.2],
            "extended_compound": [0.6, -0.1, 0.2],
        }
    )
    result = sentiment.ticker_day_sentiment(scored)
    day1 = result.loc[(result["date"] == pd.Timestamp("2021-01-04")) & (result["ticker"] == "NVDA")]
    assert day1["baseline_compound"].iloc[0] == pytest.approx((0.5 - 0.1) / 2)
    assert day1["headline_count"].iloc[0] == 2


# 8. Sector-day equal-weights active tickers, not raw headlines.
def test_sector_day_sentiment_equal_weights_active_tickers_not_headlines():
    ticker_day = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-04", "2021-01-04"]),
            "ticker": ["NVDA", "AMD"],
            "sector": ["Tech", "Tech"],
            "baseline_compound": [1.0, -1.0],
            "extended_compound": [1.0, -1.0],
            "headline_count": [100, 1],  # NVDA has far more headlines than AMD
        }
    )
    result = sentiment.sector_day_sentiment(
        ticker_day, trading_dates=pd.to_datetime(["2021-01-04"]), sector_universe={"Tech": 5}
    )
    row = result.loc[result["sector"] == "Tech"].iloc[0]
    # Equal-weight across the two ACTIVE tickers -> mean(1.0, -1.0) == 0.0, not
    # headline-weighted (which a naive "average every headline" would give,
    # dominated by NVDA's 100 headlines).
    assert row["baseline_compound"] == pytest.approx(0.0)
    assert row["active_ticker_count"] == 2
    assert row["headline_count"] == 101


# 9 & 10. No news stays NaN; never filled with 0/neutral or carried forward.
def test_sector_day_sentiment_no_news_day_is_nan_not_zero_and_not_carried_forward():
    ticker_day = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-04"]),
            "ticker": ["NVDA"],
            "sector": ["Tech"],
            "baseline_compound": [0.8],
            "extended_compound": [0.8],
            "headline_count": [1],
        }
    )
    dates = pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-06"])
    result = sentiment.sector_day_sentiment(
        ticker_day, trading_dates=dates, sector_universe={"Tech": 5}
    )

    no_news = result.loc[result["date"] > pd.Timestamp("2021-01-04")]
    assert no_news["baseline_compound"].isna().all()  # NaN, not 0.0
    assert no_news["extended_compound"].isna().all()
    assert (no_news["headline_count"] == 0).all()
    assert (no_news["active_ticker_count"] == 0).all()
    assert (no_news["coverage_ratio"] == 0.0).all()
    # Not carried forward from 2021-01-04's 0.8 value.
    assert not no_news["baseline_compound"].eq(0.8).any()


# 11. Expanding z-score uses only strictly-prior observations (no look-ahead).
def test_expanding_standardize_uses_only_strictly_prior_observations():
    dates = pd.date_range("2021-01-01", periods=70, freq="D")
    rng = np.random.default_rng(0)
    values = rng.normal(0, 1, size=70)
    df = pd.DataFrame({"date": dates, "sector": "Tech", "value": values})
    z = sentiment.expanding_standardize(df, "value", min_periods=60)

    t = 65
    history = df["value"].iloc[:t]  # strictly before t
    manual_z = (df["value"].iloc[t] - history.mean()) / history.std(ddof=1)
    assert z.iloc[t] == pytest.approx(manual_z)

    # Changing a FUTURE value must not change an earlier z-score.
    mutated = df.copy()
    mutated.loc[69, "value"] = 999.0
    z_mutated = sentiment.expanding_standardize(mutated, "value", min_periods=60)
    assert z_mutated.iloc[65] == pytest.approx(z.iloc[65])


def test_expanding_standardize_does_not_leak_across_sectors():
    dates = pd.date_range("2021-01-01", periods=10, freq="D")
    df = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "sector": ["A"] * 10 + ["B"] * 10,
            "value": [0.1, 0.2, -0.1, 0.3, 0.0, 0.15, -0.2, 0.25, 0.05, 0.1] + [100.0] * 10,
        }
    )
    z = sentiment.expanding_standardize(df, "value", min_periods=3)
    a_z = z.loc[df["sector"] == "A"]
    assert a_z.abs().max() < 10  # not blown up by sector B's huge constant values


# 12. Expanding standardisation requires exactly 60 non-missing prior observations.
def test_expanding_standardize_requires_60_prior_observations():
    dates = pd.date_range("2021-01-01", periods=65, freq="D")
    df = pd.DataFrame({"date": dates, "sector": "Tech", "value": np.arange(65, dtype=float)})
    z = sentiment.expanding_standardize(df, "value", min_periods=60)
    assert z.iloc[:60].isna().all()
    assert z.iloc[60:].notna().all()


def test_classify_sentiment_band_thresholds_and_nan_passthrough():
    z = pd.Series([-2.0, -1.0, 0.0, 1.0, 2.0, np.nan])
    band = sentiment.classify_sentiment_band(z)
    assert band.tolist()[:5] == ["extreme_fear", "fear", "neutral", "greed", "extreme_greed"]
    assert pd.isna(band.iloc[5])  # missing z stays unclassified, never "neutral"


# 13. The official index never contains 2020 rows.
def test_assert_application_period_only_rejects_2020_and_accepts_2021_2023():
    with pytest.raises(AssertionError, match="2020-12-31"):
        sentiment.assert_application_period_only(
            pd.Series(pd.to_datetime(["2020-12-31", "2021-01-04"])), context="test index"
        )
    sentiment.assert_application_period_only(  # must not raise
        pd.Series(pd.to_datetime(["2021-01-01", "2023-12-31"])), context="test index"
    )


# 14. Both sentiment figures reject a 2020-contaminated data source.
def test_figure_6_rejects_2020_contaminated_sector_index():
    dates = pd.date_range("2021-01-01", periods=61, freq="D")
    sector_index = pd.DataFrame(
        {
            "date": [*list(dates), pd.Timestamp("2020-06-01")],
            "sector": "Tech",
            "extended_z_expanding": np.linspace(-1, 1, 62),
        }
    )
    with pytest.raises(AssertionError, match="2020"):
        run_part_b.plot_extended_sector_sentiment_index(sector_index)


# ============================================================================
# Figure 6 bold line: rolling mean of the latest 21 VALID observations must
# stay missing on exactly the dates the daily index itself is missing - no
# fill, no forward/backward fill, no interpolation, no reach-through-the-gap.
# ============================================================================


def test_rolling_mean_of_valid_observations_never_fills_or_carries_forward_missing_days():
    # 50 days, all valid except index 30, so the gap sits well AFTER the
    # rolling mean has already started producing real values (first valid
    # rolling value is index 20, the 21st observation) - otherwise a
    # forward-fill probe has nothing valid to carry forward from and the test
    # would not actually exercise the carry-forward guard.
    dates = pd.date_range("2021-01-01", periods=50, freq="D")
    values = pd.Series(np.arange(50, dtype=float), index=dates)
    values.iloc[30] = np.nan  # one no-news day, with plenty of valid history either side

    rolling = run_part_b._rolling_mean_of_valid_observations(values)

    # 1) the NaN day's rolling mean is still NaN.
    assert pd.isna(rolling.iloc[30])

    # 2) not filled with 0.
    assert not (rolling.dropna() == 0).any()

    # 3) no carry-forward: index 29 has a real rolling value, so a
    #    forward-fill WOULD have produced a non-missing value at index 30 by
    #    reusing it; this function must not do that.
    assert pd.notna(rolling.iloc[29])  # sanity: there is something to carry forward
    hypothetical_ffill = rolling.ffill()
    assert pd.notna(hypothetical_ffill.iloc[30])  # ffill *would* have filled it
    assert pd.isna(rolling.iloc[30])  # but this function does not carry forward

    # 4) a position with 21 valid observations (current + history) gets a real
    #    value: days 0..20 inclusive = 21 calendar days, all valid (gap is at 30).
    valid_window = values.iloc[:21]
    assert len(valid_window) == 21 and valid_window.notna().all()
    assert rolling.iloc[20] == pytest.approx(valid_window.mean())

    # Day 19 has a valid OWN value but only 20 valid observations so far
    # (fewer than min_periods=21) - still NaN, not a partial-window value.
    assert pd.notna(values.iloc[19])
    assert pd.isna(rolling.iloc[19])

    # 5) missing stays missing everywhere the daily series is missing.
    assert rolling.loc[values.isna()].isna().all()


def test_figure_6_bold_line_stays_missing_wherever_daily_index_is_missing(monkeypatch, tmp_path):
    """Intercept the actual rendered bold-line data (not just the pure
    function) and confirm the no-news gap is never bridged in the figure.
    """
    import matplotlib.pyplot as plt

    dates = pd.date_range("2021-01-01", periods=40, freq="D")
    rows = []
    for sector in ["A", "B"]:
        z = np.sin(np.arange(40) / 3.0)
        z[10] = np.nan  # a no-news day, well inside a run of otherwise-valid days
        for date_value, z_value in zip(dates, z):
            rows.append({"date": date_value, "sector": sector, "extended_z_expanding": z_value})
    sector_index = pd.DataFrame(rows)

    monkeypatch.setattr(run_part_b, "FIGURES", tmp_path)
    captured = {}
    original_close = plt.close

    def _capture_instead_of_close(*args, **kwargs):
        captured["fig"] = plt.gcf()

    monkeypatch.setattr(plt, "close", _capture_instead_of_close)
    try:
        run_part_b.apply_theme()
        run_part_b.plot_extended_sector_sentiment_index(sector_index)
    finally:
        monkeypatch.setattr(plt, "close", original_close)

    fig = captured["fig"]
    ax = fig.axes[0]  # sector "A" (sectors are sorted alphabetically)
    thin_line, bold_line, *_ = ax.get_lines()
    original_close(fig)

    assert np.isnan(thin_line.get_ydata()[10])
    assert np.isnan(bold_line.get_ydata()[10])


def test_figure_7_rejects_2020_contaminated_sector_index():
    dates = pd.date_range("2021-01-01", periods=61, freq="D")
    sector_index = pd.DataFrame(
        {
            "date": [*list(dates), pd.Timestamp("2020-06-01")],
            "sector": "Tech",
            "baseline_z_expanding": 0.0,
            "extended_z_expanding": 0.0,
        }
    )
    sector_comparison = pd.DataFrame({"scope": ["Tech"], "custom_term_hit_rate": [0.5]})
    with pytest.raises(AssertionError, match="2020"):
        run_part_b.plot_baseline_vs_extended_sentiment(sector_index, sector_comparison)


# 15. sector_sentiment_index.csv schema (function contract, plus the real file if built).
def test_add_expanding_zscores_produces_the_required_sector_sentiment_index_schema():
    dates = pd.date_range("2021-01-01", periods=65, freq="D")
    sector_day = pd.DataFrame(
        {
            "date": dates,
            "sector": "Tech",
            "baseline_compound": np.linspace(-1, 1, 65),
            "extended_compound": np.linspace(-1, 1, 65),
            "baseline_score_100": np.linspace(0, 100, 65),
            "extended_score_100": np.linspace(0, 100, 65),
            "headline_count": 1,
            "active_ticker_count": 1,
            "sector_universe_size": 5,
            "coverage_ratio": 0.2,
        }
    )
    result = sentiment.add_expanding_zscores(sector_day, min_periods=60)
    assert result.columns.tolist() == REQUIRED_SECTOR_INDEX_COLUMNS


def test_required_sector_sentiment_index_csv_has_schema_and_no_2020_rows():
    path = ROOT / "results" / "data" / "sector_sentiment_index.csv"
    assert path.exists(), f"Required input is missing: {path}"
    df = pd.read_csv(path, parse_dates=["date"])
    missing = set(REQUIRED_SECTOR_INDEX_COLUMNS) - set(df.columns)
    assert not missing, f"missing required columns: {missing}"
    assert df["date"].min() >= pd.Timestamp("2021-01-01")
    assert df["date"].max() <= pd.Timestamp("2023-12-31")


# ============================================================================
# Overall definition unification: sector_sentiment_model_comparison.csv's
# "Overall" row and Figure 7's Overall panel must both come from
# sentiment.daily_overall_sentiment_aggregate (equal-sector-weight per date),
# never from pooling every (date, sector) row together.
# ============================================================================


def _hand_built_sector_index() -> pd.DataFrame:
    """3 sectors x 3 dates, uneven coverage (sector C missing on day 2), so a
    naive pool-everything Overall would differ from the correct daily,
    equal-sector-weight aggregate.
    """
    columns = [
        "date", "sector", "baseline_compound", "extended_compound",
        "baseline_z_expanding", "extended_z_expanding", "headline_count", "coverage_ratio",
    ]
    rows = [
        ("2021-01-04", "A", 0.2, 0.4, 1.0, 2.0, 3, 0.6),
        ("2021-01-04", "B", -0.2, -0.2, -1.0, -1.0, 2, 0.4),
        ("2021-01-04", "C", 0.6, 0.6, 2.0, 2.0, 1, 0.2),
        ("2021-01-05", "A", 0.0, 0.1, 0.0, 0.6, 2, 0.4),
        ("2021-01-05", "B", 0.4, 0.4, 1.6, 1.6, 4, 0.8),
        ("2021-01-05", "C", np.nan, np.nan, np.nan, np.nan, 0, 0.0),  # no news
        ("2021-01-06", "A", -0.3, -0.6, -1.2, -2.4, 1, 0.2),
        ("2021-01-06", "B", 0.1, 0.1, 0.4, 0.4, 3, 0.6),
        ("2021-01-06", "C", -0.5, -0.5, -2.0, -2.0, 2, 0.4),
    ]
    df = pd.DataFrame(rows, columns=columns)
    df["date"] = pd.to_datetime(df["date"])
    df["baseline_band"] = sentiment.classify_sentiment_band(df["baseline_z_expanding"])
    df["extended_band"] = sentiment.classify_sentiment_band(df["extended_z_expanding"])
    return df


def test_daily_overall_aggregate_averages_only_sectors_with_data_that_day():
    sector_index = _hand_built_sector_index()
    agg = sentiment.daily_overall_sentiment_aggregate(sector_index)

    # 2021-01-05: sector C is NaN, so the aggregate must average A and B only.
    row = agg.loc[agg["date"] == pd.Timestamp("2021-01-05")].iloc[0]
    assert row["baseline_compound"] == pytest.approx((0.0 + 0.4) / 2)
    assert row["extended_compound"] == pytest.approx((0.1 + 0.4) / 2)
    assert row["baseline_z_expanding"] == pytest.approx((0.0 + 1.6) / 2)
    assert row["extended_z_expanding"] == pytest.approx((0.6 + 1.6) / 2)


def test_sector_sentiment_model_comparison_overall_matches_hand_computed_daily_aggregate():
    sector_index = _hand_built_sector_index()
    agg = sentiment.daily_overall_sentiment_aggregate(sector_index)

    manual_mean_diff = (agg["extended_compound"] - agg["baseline_compound"]).mean()
    manual_mean_abs_diff = (agg["extended_compound"] - agg["baseline_compound"]).abs().mean()
    manual_pearson = agg["baseline_compound"].corr(agg["extended_compound"], method="pearson")
    manual_spearman = agg["baseline_compound"].corr(agg["extended_compound"], method="spearman")
    z_diff = agg["extended_z_expanding"] - agg["baseline_z_expanding"]
    manual_z_mean_diff = z_diff.mean()
    manual_z_mean_abs_diff = z_diff.abs().mean()
    manual_z_pearson = agg["baseline_z_expanding"].corr(
        agg["extended_z_expanding"], method="pearson"
    )
    manual_z_spearman = agg["baseline_z_expanding"].corr(
        agg["extended_z_expanding"], method="spearman"
    )

    headlines = pd.DataFrame({"sector": ["A", "B", "C"], "custom_term_hit": [True, False, True]})
    cmp_tbl = sentiment.sector_sentiment_model_comparison_table(sector_index, headlines)
    overall = cmp_tbl.loc[cmp_tbl["scope"] == "Overall"].iloc[0]

    assert overall["raw_mean_difference"] == pytest.approx(manual_mean_diff)
    assert overall["raw_mean_abs_difference"] == pytest.approx(manual_mean_abs_diff)
    assert overall["raw_pearson_corr"] == pytest.approx(manual_pearson)
    assert overall["raw_spearman_corr"] == pytest.approx(manual_spearman)
    assert overall["z_mean_difference"] == pytest.approx(manual_z_mean_diff)
    assert overall["z_mean_abs_difference"] == pytest.approx(manual_z_mean_abs_diff)
    assert overall["z_pearson_corr"] == pytest.approx(manual_z_pearson)
    assert overall["z_spearman_corr"] == pytest.approx(manual_z_spearman)

    # A naive pool-every-row Overall would give a DIFFERENT answer on this
    # deliberately uneven fixture - guard against silently reverting to that.
    pooled_diff = sector_index["extended_compound"] - sector_index["baseline_compound"]
    assert overall["raw_mean_difference"] != pytest.approx(pooled_diff.mean())


def test_sector_sentiment_model_comparison_per_sector_rows_unaffected_by_overall_fix():
    sector_index = _hand_built_sector_index()
    headlines = pd.DataFrame({"sector": ["A", "B", "C"], "custom_term_hit": [True, False, True]})
    cmp_tbl = sentiment.sector_sentiment_model_comparison_table(sector_index, headlines)

    for sector in ["A", "B", "C"]:
        rows = sector_index.loc[sector_index["sector"] == sector]
        manual_pearson = rows["baseline_compound"].corr(rows["extended_compound"])
        table_row = cmp_tbl.loc[cmp_tbl["scope"] == sector].iloc[0]
        assert table_row["raw_pearson_corr"] == pytest.approx(manual_pearson)


def test_figure_7_overall_panel_plots_the_exact_daily_aggregate_values(monkeypatch, tmp_path):
    """Intercept the actual rendered line data in Figure 7's first ("Overall
    aggregate") panel and check it against sentiment.daily_overall_sentiment_
    aggregate directly - not just that the two happen to be reproducible.
    """
    import matplotlib.pyplot as plt

    sector_index = _hand_built_sector_index()
    # A 4th sector with a much larger hit rate than A/B/C, so it is
    # unambiguously picked as "highest custom-term hit rate" and does not
    # collide with the Overall panel under test.
    extra = sector_index.loc[sector_index["sector"] == "A"].copy()
    extra["sector"] = "D"
    sector_index = pd.concat([sector_index, extra], ignore_index=True)
    headlines = pd.DataFrame(
        {"sector": ["A", "B", "C", "D"], "custom_term_hit": [False, False, False, True]}
    )
    sector_comparison = sentiment.sector_sentiment_model_comparison_table(sector_index, headlines)
    expected_aggregate = sentiment.daily_overall_sentiment_aggregate(sector_index)

    monkeypatch.setattr(run_part_b, "FIGURES", tmp_path)

    captured = {}
    original_close = plt.close

    def _capture_instead_of_close(*args, **kwargs):
        captured["fig"] = plt.gcf()

    monkeypatch.setattr(plt, "close", _capture_instead_of_close)
    try:
        run_part_b.apply_theme()
        run_part_b.plot_baseline_vs_extended_sentiment(sector_index, sector_comparison)
    finally:
        monkeypatch.setattr(plt, "close", original_close)

    fig = captured["fig"]
    overall_ax = fig.axes[0]
    lines_by_label = {line.get_label(): line for line in overall_ax.get_lines()}
    plotted_baseline = lines_by_label["Baseline FinVADER"].get_ydata()
    plotted_extended = lines_by_label["Extended FinVADER"].get_ydata()

    expected_baseline = expected_aggregate["baseline_z_expanding"].to_numpy()
    expected_extended = expected_aggregate["extended_z_expanding"].to_numpy()
    np.testing.assert_allclose(plotted_baseline, expected_baseline)
    np.testing.assert_allclose(plotted_extended, expected_extended)
    original_close(fig)



# ============================================================================
# Figures 9 and 10: report-only market-wide and sector-ranking sentiment
# exhibits. Both must reuse the shared Overall aggregate and the shared
# rolling-mean definition rather than re-deriving either.
# ============================================================================


def _capture_figure(monkeypatch, tmp_path, plot_callable, *args):
    """Render a figure, keeping the matplotlib Figure alive for inspection."""
    import matplotlib.pyplot as plt

    monkeypatch.setattr(run_part_b, "FIGURES", tmp_path)
    captured = {}
    original_close = plt.close

    def _capture_instead_of_close(*_args, **_kwargs):
        captured["fig"] = plt.gcf()

    monkeypatch.setattr(plt, "close", _capture_instead_of_close)
    try:
        run_part_b.apply_theme()
        path = plot_callable(*args)
    finally:
        monkeypatch.setattr(plt, "close", original_close)
    return captured["fig"], path, original_close


def _load_required_sector_index() -> pd.DataFrame:
    path = ROOT / "results" / "data" / "sector_sentiment_index.csv"
    assert path.exists(), f"Required input is missing: {path}"
    return pd.read_csv(path, parse_dates=["date"])


def _uneven_sector_index() -> pd.DataFrame:
    """Three sectors over four dates, with deliberately uneven coverage.

    Sector C has no compound on day 2 and no z on day 3, so the compound and
    z aggregates must use different validity masks on those days.
    """
    rows = [
        ("2021-01-04", "A", 0.20, 1.0), ("2021-01-04", "B", -0.20, -1.0),
        ("2021-01-04", "C", 0.60, 2.0),
        ("2021-01-05", "A", 0.00, 0.0), ("2021-01-05", "B", 0.40, 1.6),
        ("2021-01-05", "C", np.nan, np.nan),
        ("2021-01-06", "A", -0.30, -1.2), ("2021-01-06", "B", 0.10, 0.4),
        ("2021-01-06", "C", -0.50, np.nan),
        ("2021-01-07", "A", 0.10, 0.5), ("2021-01-07", "B", 0.30, 1.0),
        ("2021-01-07", "C", 0.20, 0.9),
        # Every sector has a compound but no z yet - the warm-up case, where
        # the aggregate z must stay missing rather than become a zero bar.
        ("2021-01-08", "A", 0.10, np.nan), ("2021-01-08", "B", 0.20, np.nan),
        ("2021-01-08", "C", 0.30, np.nan),
    ]
    frame = pd.DataFrame(
        rows, columns=["date", "sector", "extended_compound", "extended_z_expanding"]
    )
    frame["date"] = pd.to_datetime(frame["date"])
    # daily_overall_sentiment_aggregate also needs the baseline columns.
    frame["baseline_compound"] = frame["extended_compound"]
    frame["baseline_z_expanding"] = frame["extended_z_expanding"]
    return frame


def test_figure_9_plots_the_shared_overall_aggregate_with_per_column_masks(
    monkeypatch, tmp_path
):
    sector_index = _uneven_sector_index()
    expected = sentiment.daily_overall_sentiment_aggregate(sector_index).sort_values("date")

    # Hand-checked: day 2 compound averages A and B only (C is missing), and
    # day 3 z averages A and B only even though C has a compound that day.
    day2 = expected.loc[expected["date"] == pd.Timestamp("2021-01-05")].iloc[0]
    assert day2["extended_compound"] == pytest.approx((0.00 + 0.40) / 2)
    day3 = expected.loc[expected["date"] == pd.Timestamp("2021-01-06")].iloc[0]
    assert day3["extended_compound"] == pytest.approx((-0.30 + 0.10 - 0.50) / 3)
    assert day3["extended_z_expanding"] == pytest.approx((-1.2 + 0.4) / 2)

    fig, _path, close = _capture_figure(
        monkeypatch, tmp_path, run_part_b.plot_market_wide_sentiment_index, sector_index
    )
    upper, lower = fig.axes[0], fig.axes[1]
    daily_line = next(
        line for line in upper.get_lines()
        if line.get_label() == "Daily market sentiment score"
    )
    np.testing.assert_allclose(
        daily_line.get_ydata(),
        run_part_b.market_score_100(expected["extended_compound"]).to_numpy(),
    )
    bar_heights = np.array([patch.get_height() for patch in lower.patches])
    np.testing.assert_allclose(
        bar_heights, expected["extended_z_expanding"].to_numpy(), equal_nan=True
    )
    close(fig)


@pytest.mark.parametrize(
    "compound, expected_score", [(-1.0, 0.0), (0.0, 50.0), (1.0, 100.0), (0.5, 75.0)]
)
def test_market_score_100_is_a_linear_relabelling_of_the_compound_score(
    compound, expected_score
):
    result = run_part_b.market_score_100(pd.Series([compound]))
    assert result.iloc[0] == pytest.approx(expected_score)


def test_figure_9_rolling_mean_uses_valid_observations_and_keeps_gaps_blank(
    monkeypatch, tmp_path
):
    dates = pd.bdate_range("2021-01-04", periods=40)
    rows = []
    for index, date in enumerate(dates):
        compound = 0.02 * ((index % 5) - 2)
        # One completely missing market day, mid-sample.
        missing = index == 25
        rows.append(
            {
                "date": date, "sector": "A",
                "extended_compound": np.nan if missing else compound,
                "extended_z_expanding": np.nan if missing else compound * 5,
            }
        )
    frame = pd.DataFrame(rows)
    frame["baseline_compound"] = frame["extended_compound"]
    frame["baseline_z_expanding"] = frame["extended_z_expanding"]

    fig, _path, close = _capture_figure(
        monkeypatch, tmp_path, run_part_b.plot_market_wide_sentiment_index, frame
    )
    upper = fig.axes[0]
    daily = next(
        line for line in upper.get_lines()
        if line.get_label() == "Daily market sentiment score"
    ).get_ydata()
    rolling = next(
        line for line in upper.get_lines()
        if "rolling mean" in line.get_label()
    ).get_ydata()

    daily_series = pd.Series(daily)
    rolling_series = pd.Series(rolling)
    # The missing day is missing in both series - no rolling bleed.
    assert np.isnan(daily[25])
    assert np.isnan(rolling[25])
    assert rolling_series[daily_series.isna()].isna().all()
    # The window advances by valid observations, matching the shared helper.
    expected = run_part_b._rolling_mean_of_valid_observations(daily_series)
    np.testing.assert_allclose(rolling, expected.to_numpy(), equal_nan=True)
    # min_periods=21 valid observations before the first value appears.
    assert rolling_series.notna().sum() == max(0, daily_series.notna().sum() - 20)
    close(fig)


def test_figure_9_anomaly_bars_colour_by_sign_and_never_fill_missing_days(
    monkeypatch, tmp_path
):
    import matplotlib.colors as mcolors

    sector_index = _uneven_sector_index()
    expected = sentiment.daily_overall_sentiment_aggregate(sector_index).sort_values("date")

    fig, _path, close = _capture_figure(
        monkeypatch, tmp_path, run_part_b.plot_market_wide_sentiment_index, sector_index
    )
    lower = fig.axes[1]
    heights = np.array([patch.get_height() for patch in lower.patches])
    colours = [mcolors.to_hex(patch.get_facecolor()).upper() for patch in lower.patches]

    positive = mcolors.to_hex(run_part_b.SENTIMENT_POSITIVE_COLOR).upper()
    negative = mcolors.to_hex(run_part_b.SENTIMENT_NEGATIVE_COLOR).upper()
    assert positive != negative
    for height, colour in zip(heights, colours):
        if np.isnan(height):
            continue
        if height > 0:
            assert colour == positive
        elif height < 0:
            assert colour == negative

    # Missing z stays NaN rather than becoming a zero-height bar.
    np.testing.assert_allclose(
        heights, expected["extended_z_expanding"].to_numpy(), equal_nan=True
    )
    assert np.isnan(heights).any()
    close(fig)


def test_figure_9_axes_ticks_and_plotted_dates_stay_inside_the_real_data_period(
    monkeypatch, tmp_path
):
    import matplotlib.dates as mdates

    sector_index = _load_required_sector_index()
    aggregate = sentiment.daily_overall_sentiment_aggregate(sector_index).sort_values("date")
    start_date = pd.Timestamp("2021-01-04")
    end_date = pd.Timestamp("2023-12-29")
    assert aggregate["date"].min() == start_date
    assert aggregate["date"].max() == end_date

    fig, _path, close = _capture_figure(
        monkeypatch, tmp_path, run_part_b.plot_market_wide_sentiment_index, sector_index
    )
    fig.canvas.draw()
    upper, lower = fig.axes[0], fig.axes[1]
    start_num = mdates.date2num(start_date.to_pydatetime())
    end_num = mdates.date2num(end_date.to_pydatetime())
    date_tolerance = 1e-6

    for ax in (upper, lower):
        left, right = ax.get_xlim()
        assert left >= start_num - date_tolerance
        assert right <= end_num + date_tolerance

    visible_date_ticks = [
        label.get_text()
        for ax in (upper, lower)
        for label in ax.get_xticklabels()
        if label.get_visible() and label.get_text()
    ]
    assert visible_date_ticks
    assert not any("2020" in label or "2024" in label for label in visible_date_ticks)

    dated_lines = [
        line
        for line in upper.get_lines()
        if line.get_label()
        in {
            "Daily market sentiment score",
            "21-observation rolling mean (display only)",
        }
    ]
    assert len(dated_lines) == 2
    for line in dated_lines:
        plotted_dates = pd.to_datetime(line.get_xdata())
        assert plotted_dates.min() >= start_date
        assert plotted_dates.max() <= end_date

    bar_centres = np.array(
        [patch.get_x() + patch.get_width() / 2 for patch in lower.patches]
    )
    assert bar_centres.size == len(aggregate)
    assert bar_centres.min() >= start_num - date_tolerance
    assert bar_centres.max() <= end_num + date_tolerance
    close(fig)


def test_figures_9_and_10_reject_dates_outside_the_application_period():
    contaminated = _uneven_sector_index()
    contaminated.loc[0, "date"] = pd.Timestamp("2020-12-31")
    with pytest.raises(AssertionError, match="2020"):
        run_part_b.plot_market_wide_sentiment_index(contaminated)
    with pytest.raises(AssertionError, match="2020"):
        run_part_b.plot_sector_sentiment_ranking(contaminated)


def test_sector_ranking_table_matches_hand_computed_pandas_statistics():
    sector_index = _uneven_sector_index()
    ranking = run_part_b.sector_sentiment_ranking_table(sector_index)

    valid = sector_index.dropna(subset=["extended_compound"])
    for sector in ["A", "B", "C"]:
        rows = valid.loc[valid["sector"] == sector, "extended_compound"]
        result = ranking.loc[ranking["sector"] == sector].iloc[0]
        assert result["mean_extended_compound"] == pytest.approx(rows.mean())
        assert result["median_extended_compound"] == pytest.approx(rows.median())
        assert result["n_valid_sector_days"] == len(rows)

    # A and B have a compound on all five dates; C is missing one, so C must
    # be averaged over four days rather than being padded back up to five.
    counts = ranking.set_index("sector")["n_valid_sector_days"]
    assert counts["A"] == 5 and counts["B"] == 5
    assert counts["C"] == 4
    # Descending by mean.
    means = ranking["mean_extended_compound"].to_numpy()
    assert (np.diff(means) <= 0).all()


def test_sector_ranking_is_not_weighted_by_headline_count():
    """Adding headline counts must not change the ranking statistics."""
    sector_index = _uneven_sector_index()
    plain = run_part_b.sector_sentiment_ranking_table(sector_index)

    weighted_input = sector_index.copy()
    weighted_input["headline_count"] = np.where(
        weighted_input["sector"] == "A", 1000, 1
    )
    weighted_input["active_ticker_count"] = weighted_input["headline_count"]
    with_counts = run_part_b.sector_sentiment_ranking_table(weighted_input)

    pdt.assert_frame_equal(plain, with_counts)


def test_figure_10_plots_all_ten_real_sectors_in_ranked_order(monkeypatch, tmp_path):
    sector_index = _load_required_sector_index()
    ranking = run_part_b.sector_sentiment_ranking_table(sector_index)
    assert len(ranking) == 10

    fig, _figure_path, close = _capture_figure(
        monkeypatch, tmp_path, run_part_b.plot_sector_sentiment_ranking, sector_index
    )
    ax = fig.axes[0]
    # barh-style layout: highest mean at the top, so labels read in reverse.
    plotted_sectors = [label.get_text() for label in ax.get_yticklabels()]
    assert plotted_sectors == list(ranking["sector"])[::-1]

    # Select by label: collections also holds the stem LineCollection.
    by_label = {c.get_label(): c for c in ax.collections}
    np.testing.assert_allclose(
        by_label["Mean"].get_offsets()[:, 0],
        ranking["mean_extended_compound"].to_numpy()[::-1],
    )
    np.testing.assert_allclose(
        by_label["Median"].get_offsets()[:, 0],
        ranking["median_extended_compound"].to_numpy()[::-1],
    )
    assert ax.get_xlabel() == "Extended FinVADER compound score"
    close(fig)


@pytest.mark.parametrize(
    "plot_callable",
    [
        run_part_b.plot_market_wide_sentiment_index,
        run_part_b.plot_sector_sentiment_ranking,
    ],
)
def test_figures_9_and_10_use_extended_only_source_notes(
    monkeypatch, tmp_path, plot_callable
):
    sector_index = _load_required_sector_index()
    fig, _path, close = _capture_figure(
        monkeypatch, tmp_path, plot_callable, sector_index
    )
    figure_text = "\n".join(text.get_text() for text in fig.texts)
    assert "Extended FinVADER" in figure_text
    assert "30-term custom lexicon" in figure_text
    assert "2021–2023" in figure_text
    assert "baseline vs Extended" not in figure_text
    close(fig)


def test_figures_9_and_10_are_written_with_the_exact_required_filenames():
    figures = ROOT / "results" / "figures"
    for name in [
        "figure_9_market_wide_sentiment_index.png",
        "figure_10_sector_sentiment_ranking.png",
    ]:
        path = figures / name
        assert path.exists(), f"Required output is missing: {path}"
        assert path.stat().st_size > 0, f"Required output is empty: {path}"


def test_real_figure_9_and_10_inputs_are_restricted_to_2021_2023():
    sector_index = _load_required_sector_index()
    expected_start = pd.Timestamp("2021-01-04")
    expected_end = pd.Timestamp("2023-12-29")
    assert sector_index["date"].min() == expected_start
    assert sector_index["date"].max() == expected_end

    aggregate = sentiment.daily_overall_sentiment_aggregate(sector_index)
    assert aggregate["date"].min() == expected_start
    assert aggregate["date"].max() == expected_end
    assert not (aggregate["date"].dt.year.isin([2020, 2024])).any()
