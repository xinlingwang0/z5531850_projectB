# FINS3645 Part B AI Instructions

## Project Identity

This folder is `z5531850_projectB`, the Part B project for RiskBridge Funds.
RiskBridge Funds is the investment app developed across the project. It offers
systematically managed equity, crypto, and combined funds, and helps an investor
compare fund performance, risk, holdings, and market sentiment.

Part B covers Data Factory Floor Stations 3-4:

- Station 3: out-of-sample portfolio construction, FinVADER sentiment analysis,
  and sentiment integration.
- Station 4: a deployed Streamlit investment app.

Read `PROJECT_BRIEF.md` first, followed by `context/DATA_GUIDE.md` and
`context/project_context.md`. The project brief is the final authority if any
instruction conflicts.

## Project Goals

Build a complete and reproducible Part B workflow that:

- constructs equity-only, crypto-only, and combined equity-crypto funds;
- compares several portfolio methods using walk-forward out-of-sample tests;
- produces a sector news-sentiment index with FinVADER;
- integrates lagged equity sentiment into a fund and compares it with the base
  fund;
- generates the required tables, figures, and app data; and
- presents the results through a clear Streamlit investor journey.

All claims must be supported by reproducible outputs. Do not invent values,
citations, test results, or economic explanations.

## File Responsibilities

- `src/etl.py`: load and clean the course datasets through `src/data_access.py`.
- `src/features.py`: calculate returns, align calendars, and assemble headlines
  by equity trading day.
- `src/portfolios.py`: portfolio methods, walk-forward backtests, weights, fund
  returns, and performance metrics.
- `src/sentiment.py`: FinVADER scoring, ticker-day aggregation, coverage checks,
  and sector sentiment indices.
- `src/fusion.py`: lagged sentiment signals, portfolio tilts, and
  base-versus-sentiment comparison.
- `scripts/run_part_b.py`: run the complete build and save all results.
- `streamlit_app.py`: load precomputed results and present the app.
- `tests/`: unit, data-contract, and smoke tests.

Keep reusable logic in `src/`, orchestration in `scripts/`, generated files in
`results/`, report files in `report/`, and AI-use records in `ai/`. Use
project-relative paths and small testable functions.

Do not edit the provided `src/data_access.py`, files under `context/`, or
`scripts/check_handin.py` unless explicitly instructed.

## Data And Feature Rules

Load equity prices, crypto prices, and headlines through `src/data_access.py`.

Use `adjClose` and simple daily returns. Sort by `ticker` and `date`, then
calculate returns within each ticker using `pct_change(fill_method=None)`.

Cap the crypto sample at `2023-12-31`.

Calculate equity and crypto returns on their own calendars before combining
them. Build the combined return panel by aligning already-calculated crypto
returns to the equity trading calendar. Never merge price levels first and
calculate returns afterward.

Use 252 periods per year for equity and equity-calendar combined funds, and 365
for crypto-only funds on the crypto calendar.

Normalize news and price dates before alignment. Map each headline to the same
equity trading day, or to the next equity trading day when the date is not a
trading day.

Deduplicate prices on `ticker, date` and headlines on `ticker, date, title`.

Preserve the original headline text for sentiment scoring.

Document keys, sample periods, missing-value policies, and important data checks
for every derived dataset.

## Portfolio And Backtest Rules

Create equity-only, crypto-only, and combined fund families. Include an
equal-weight benchmark and multiple portfolio methods such as minimum variance,
risk parity, and maximum Sharpe.

The backtest must be walk-forward and out of sample:

- Use an initial estimation window before the first live return.
- Rebalance monthly or less often using a stated rule.
- Estimate inputs using only information available at the formation date.
- Apply each new weight vector only to later holding-period returns.
- Keep formation dates and return dates separate in the outputs.
- State the estimation window, constraints, risk-free rate, transaction-cost
  assumption, and missing-return policy.

Use long-only, fully invested weights unless another constraint is clearly
justified. Check that weights sum to one, satisfy their bounds, and differ across
methods when expected. Check optimizer success and use a documented fallback if
an optimization fails. If a SciPy optimizer fails to converge for a rebalance,
fall back to Equal Weight (1/N) across the eligible assets for that period, emit
a warning, and record the fallback in the diagnostics. Avoid look-ahead in
parameter choices or model tuning.

For each fund, calculate at least:

- growth of $1;
- annualized return;
- annualized volatility;
- Sharpe ratio;
- maximum drawdown; and
- current holdings from the most recent rebalance.

Use consistent formulas and compare funds on a common sample where required.

## FinVADER Sentiment Rules

Use the `finvader` library for all headline sentiment scoring. Do not silently
substitute `nltk.sentiment.vader`.

Record the installed `finvader` package version and the selected
`use_sentibignomics`, `use_henry`, and `indicator` settings. Before coding,
verify these option names against the installed `finvader` API and update the
implementation if the package interface differs.

Keep `finvader` in `requirements-dev.txt`, not the deployed app's
`requirements.txt`. Verify that the build Python version is compatible with the
pinned package.

Preserve headline casing, punctuation, negation, and intensifiers unless a
documented FinVADER requirement states otherwise.

The sentiment workflow must:

- score each deduplicated headline;
- aggregate headline scores to ticker-trading-day values;
- retain headline counts and coverage information;
- build a sector index by equal-weighting ticker-day scores within each sector;
  and
- state how ticker-days with no headlines are treated.

Lag the aligned sentiment signal by at least one equity trading day before it can
affect a portfolio. A weekend or Monday headline aligned to Monday must not
affect the portfolio until Tuesday or later. Test the lag directly.

## Sentiment Integration Rules

Apply sentiment only to the equity assets because the news dataset covers
equities only. Start from a base portfolio and use the lagged sentiment signal to
make a transparent, bounded weight adjustment. Preserve the portfolio
constraints and renormalize adjusted weights to sum to one.

Compare the base and sentiment-adjusted funds using the same:

- out-of-sample dates;
- rebalance dates;
- investable universe;
- metric definitions; and
- transaction-cost assumption.

Report the result honestly even if sentiment does not improve performance. Do
not tune the sentiment rule on the reported out-of-sample period.

## Required Outputs

The build must create these exact files:

- `results/data/fund_returns.csv`
- `results/data/fund_weights.csv`
- `results/data/sector_sentiment_index.csv`
- `results/tables/performance_metrics.csv`

It must also create the required report exhibits:

- performance metrics across funds and methods;
- growth-of-$1 comparison;
- drawdown for at least one fund;
- portfolio weights over time for at least one fund across methods;
- Sharpe or return-versus-risk comparison;
- sector sentiment time series; and
- a table and figure comparing the base and sentiment-adjusted funds.

Every figure must include a clear title or caption, labelled axes, units, sample
period, and legend where needed. Results used in the app, tables, and figures
must be consistent with one another.

## Streamlit App Rules

The app must allow the user to:

- compare the available funds;
- open a fund fact sheet;
- inspect growth, drawdown, risk, and current holdings;
- set an illustrative allocation across funds; and
- explore sector sentiment and the sentiment-integration result.

The deployed app must read precomputed files from `results/`. It must not run
FinVADER, optimize portfolios, or recompute the backtest. Route all
`pd.read_csv` operations through helper functions decorated with
`@st.cache_data` so Streamlit does not reload the same artifacts on every rerun.
Show a clear error if a required artifact is missing. Do not present performance
as guaranteed, and label the allocation tool as illustrative rather than trade
execution.

## Testing And Verification

Add tests for the calculations and timing rules most likely to fail, including:

- a hand-checked adjusted-close return;
- crypto calendar alignment;
- non-trading-day headline mapping;
- the one-trading-day sentiment lag;
- portfolio weight constraints;
- separation of formation and holding dates;
- performance metrics on a known return series; and
- required output filenames and columns.

Run the following from the project root before completion:

- `python tests/test_smoke.py`
- `python scripts/run_part_b.py`
- Launch `streamlit run streamlit_app.py`, inspect the app locally, then stop the
  server.
- `python scripts/check_handin.py`
- `git status`

Inspect the generated CSV files and figures as well as the command results. Fix
all `[FAIL]` messages from `check_handin.py`.

## Report And AI Workflow

Write for a financially literate, non-technical reader. Explain the fund design,
backtest assumptions, results, sentiment method, integration result, limitations,
investor journey, and three concrete recommendations. Reference and interpret
every table and figure. Trace every reported number to a reproducible output and
verify every citation.

The final economic interpretation must be written in my own words. Treat AI
output as a draft to check rather than a fact to trust.

For each substantial AI-assisted task, keep a concise record under `ai/` of:

- the task and prompt;
- what the assistant produced;
- any error, risk, or uncertainty found;
- what I changed and why; and
- how the corrected result was verified.

Do not fabricate mistakes or claim that a check was completed when it was not.

## Completion Criteria

The project is complete when the build is reproducible, required files and
exhibits are present, no-look-ahead checks pass, the Streamlit app works from
precomputed results, `check_handin.py` has no failures, and the report and AI
workflow records are complete.

When an important requirement is ambiguous, state the assumption and ask before
making a choice that materially changes the analysis.
