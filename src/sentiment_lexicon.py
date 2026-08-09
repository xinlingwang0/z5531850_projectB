"""RiskBridge's frozen 30-term custom finance lexicon.

These terms were selected from the 2020 candidate-term discovery pass
(``scripts/run_part_b.py --sentiment-candidates-only`` -> ``src/sentiment.candidate_terms``)
and scored on VADER's native -4 to +4 valence scale by 5 AI models, 2 ratings
per model (10 ratings per term), followed by frozen adjudication. Discovery
used only 2020 headlines; these 30 terms and their valences are frozen and are
not revised based on 2021-2023 application-period results (see
PROJECT_BRIEF.md and the Station 3 sentiment task instructions).

Do not add, remove, or reweight terms here based on downstream performance.
"""
from __future__ import annotations

from types import MappingProxyType

LEXICON_NAME = "RiskBridge Custom Finance Lexicon"
LEXICON_VERSION = "1.0"
DISCOVERY_PERIOD = "2020-01-01 to 2020-12-31"
APPLICATION_PERIOD = "2021-01-01 to 2023-12-31"
VALENCE_SCALE = "native VADER -4 to +4"
SCORING_PROTOCOL = (
    "5 AI models, 2 ratings per model, 10 ratings per term, followed by frozen "
    "adjudication"
)

_CUSTOM_SENTIMENT_LEXICON: dict[str, float] = {
    "jumps": 1.4,
    "jumped": 1.4,
    "selloff": -2.9,
    "sell-off": -2.9,
    "soars": 2.4,
    "soaring": 2.3,
    "surged": 2.4,
    "crashes": -3.9,
    "rallies": 2.4,
    "soared": 2.4,
    "delays": -1.7,
    "downgraded": -2.8,
    "better-than-expected": 2.9,
    "sinks": -2.2,
    "rebounding": 1.9,
    "climbs": 1.6,
    "upgraded": 2.8,
    "tumbles": -2.5,
    "plunges": -2.9,
    "comeback": 2.2,
    "layoffs": -2.8,
    "rebounds": 1.9,
    "plunged": -2.9,
    "rallied": 2.4,
    "rout": -3.6,
    "hampered": -2.0,
    "crashed": -3.9,
    "retreats": -1.5,
    "bearish": -2.4,
    "breakout": 1.7,
}

# Immutable so importers cannot accidentally add/remove/reweight terms at runtime.
CUSTOM_SENTIMENT_LEXICON: MappingProxyType[str, float] = MappingProxyType(
    _CUSTOM_SENTIMENT_LEXICON
)

N_TERMS = len(CUSTOM_SENTIMENT_LEXICON)
if N_TERMS != 30:
    raise AssertionError(
        f"CUSTOM_SENTIMENT_LEXICON must have exactly 30 frozen terms, found {N_TERMS}"
    )
