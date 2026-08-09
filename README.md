# RiskBridge Funds - FINS3645 Part B

This repository contains my FINS3645 Part B project: **RiskBridge Funds: Systematic Multi-Asset Funds with Finance-Aware Sentiment Analytics**.

Part B covers **Data Factory Floor Stations 3-4**. It constructs and evaluates systematic equity, crypto, and combined funds using walk-forward out-of-sample backtests; develops a sector news-sentiment index; tests sentiment-based portfolio adjustments; and presents the results through an investor-facing Streamlit application.

The deployed application reads precomputed artifacts from `results/`. It does not rerun portfolio optimisation or sentiment scoring during user interaction.

## Project links

- **Live Streamlit app:** [https://z5531850projectb-xons7qss6aywlvprwppgwx.streamlit.app](https://z5531850projectb-xons7qss6aywlvprwppgwx.streamlit.app/)
- **GitHub repository:** `https://github.com/xinlingwang0/z5531850_projectB`

## Project concept

RiskBridge Funds is a prototype investment application for retail investors who want to compare systematic multi-asset funds using transparent evidence about performance, risk, holdings, and market sentiment.

The Part B objective is to turn the Part A data foundation into an investable-fund prototype containing:

- equity-only, crypto-only, and combined fund families;
- walk-forward out-of-sample portfolio backtests;
- Equal Weight, Minimum Variance, Maximum Sharpe, and Hierarchical Risk Parity methods;
- fund fact sheets covering return, volatility, Sharpe ratio, drawdown, and holdings;
- a FinVADER-based sector news-sentiment index;
- lagged sentiment-based portfolio tilts;
- an interactive Streamlit investor journey;
- reproducible tables, figures, and app-ready data.

## App features

The Streamlit application provides four main investor views:

- **Fund Comparison:** compare all 12 fund-family and portfolio-method combinations using consistent out-of-sample performance metrics.
- **Fund Fact Sheet:** inspect growth of $1, drawdown, annualised return, volatility, Sharpe ratio, and current holdings for a selected fund.
- **Allocation Lab:** create an illustrative allocation across the available funds and inspect the combined historical risk-return profile.
- **Sentiment Analytics:** explore the production Extended FinVADER sector sentiment index and evaluate the base, momentum, and contrarian sentiment portfolios. The baseline-versus-extended validation is documented in the model-validation exhibit.

The allocation tool is illustrative only and does not execute trades or represent a recommendation.

## Analytical focus

This version focuses on **Out-of-Sample Multi-Asset Fund Design and Finance-Aware Sentiment Fusion**.

The analysis asks two main questions:

1. How do systematic equity, crypto, and combined funds compare when evaluated using the same look-ahead-safe walk-forward framework?
2. Does lagged, finance-aware news sentiment improve the risk-return characteristics of a base equity portfolio?

The project extension focuses on:

- comparing four portfolio construction methods across three fund families;
- preserving the different equity and crypto trading calendars and annualisation conventions;
- extending financial sentiment coverage with a custom finance-aware lexicon;
- constructing standardized sector sentiment indices with explicit coverage diagnostics;
- comparing momentum and contrarian sentiment tilts with the unadjusted base fund;
- measuring turnover, transaction costs, solver fallbacks, and backtest integrity;
- reporting mixed or negative sentiment-fusion results without retuning the out-of-sample test.

## Methodology summary

The fund universe contains equity-only, crypto-only, and combined portfolios. Each family is evaluated using Equal Weight, Minimum Variance, Maximum Sharpe, and Hierarchical Risk Parity.

Portfolio weights are formed on the final available observation of each month and become effective on the next trading observation. Equity-only and combined funds use a 252-observation estimation window and 252-day annualisation. Crypto-only funds remain on their native calendar and use a 365-observation window and 365-day annualisation.

All portfolios are long-only and fully invested. Maximum Sharpe assumes a zero risk-free rate. Performance is reported net of a transaction cost equal to 0.1% of portfolio turnover. Solver failures trigger a documented Equal Weight fallback.

News headlines are mapped to the appropriate equity trading day and scored using FinVADER. The project compares a baseline finance sentiment model with an extended custom lexicon. Sentiment signals are lagged by at least one equity trading day before they can affect portfolio weights.

The fusion test starts from the Equity-Only Minimum Variance fund. It compares the unchanged base portfolio with fixed momentum and contrarian tilts using the same rebalance dates, investable universe, return dates, and transaction-cost assumptions. The tilt directions were fixed in advance rather than selected using the reported out-of-sample results.

## Key findings

- **The Combined Maximum Sharpe fund produced the strongest risk-adjusted result across the 12 funds.** Over the 2021-2023 out-of-sample period, it recorded a 23.50% annualised return, 25.14% annualised volatility, a 0.965 Sharpe ratio, and a −25.96% maximum drawdown. Its higher return therefore came with materially greater risk than the lower-volatility funds.

- **Crypto delivered high return potential but severe downside risk.** Crypto-Only HRP recorded the highest annualised return among the tested funds at 43.00%, but it also experienced 76.95% annualised volatility and a −78.11% maximum drawdown. This result shows why return alone is insufficient when comparing investable funds.

- **Finance-aware sentiment changed headline classifications, but portfolio improvement was not consistent.** The extended lexicon affected 2,832 headlines and reclassified all 486 custom-term headlines that were exactly neutral under the baseline. In the portfolio test, the contrarian tilt modestly improved annualised return from 5.20% to 6.06% and Sharpe from 0.462 to 0.504, but it increased volatility, drawdown, and turnover. The momentum tilt underperformed the base fund. The evidence therefore does not support treating sentiment as a guaranteed source of improved performance.

These findings are historical out-of-sample results for the selected 2021-2023 sample and should not be interpreted as forecasts.

## How to run

Python 3.13 is recommended. From the project folder:

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
python scripts/run_part_b.py
python -m pytest -q
python scripts/check_handin.py
streamlit run streamlit_app.py
```

The full build reproduces the outputs under `results/`.

The deployed app only requires `requirements.txt` and loads the precomputed CSV files under `results/`. Sentiment scoring and portfolio optimisation are performed by the offline build rather than during app use.

## Folder structure

```text
PROJECT_BRIEF.md        Project brief supplied with the starter folder
README.md               This file
AGENTS.md               My AI-agent instructions
streamlit_app.py        Streamlit application entrypoint
.streamlit/             Streamlit configuration
src/                    Data, portfolio, sentiment, and fusion code
scripts/                Build and hand-in verification scripts
tests/                  Calculation, timing, data-contract, and app tests
results/                Generated data, tables, and figures
report/                 Word/PDF report files
context/                Provided data guide and project context
docs/                   Student deployment guidance
ai/                     Prompt logs and AI workflow notes
requirements.txt        Lightweight deployed-app dependencies
requirements-dev.txt    Offline build and sentiment dependencies
SUBMISSION_CHECKLIST.md Final hand-in checklist
```

## Expected outputs

The required Part B outputs include:

```text
results/data/fund_returns.csv
results/data/fund_weights.csv
results/data/sector_sentiment_index.csv
results/tables/performance_metrics.csv
```

Additional reproducible outputs include:

```text
results/data/fusion_returns.csv
results/data/fusion_weights.csv
results/data/ticker_sentiment_z.csv
results/tables/current_holdings.csv
results/tables/fusion_before_vs_after.csv
results/tables/fund_backtest_diagnostics.csv
results/tables/fusion_diagnostics.csv
results/tables/sector_sentiment_model_comparison.csv
results/figures/
report/report.docx
report/report.pdf
ai/prompt_log.md
```

The figures cover fund growth, drawdown, portfolio weights, risk-return comparisons, backtest diagnostics, sector sentiment, lexicon comparisons, and the base-versus-sentiment fusion result.

## AI use

AI assistance was used to help interpret the project requirements, review the portfolio and sentiment workflow, identify look-ahead and validation risks, design tests, troubleshoot implementation issues, and prepare deployment documentation.

I reviewed the generated code and suggestions, tested timing and portfolio constraints, checked the generated tables and figures, and retained responsibility for the final economic interpretation and written analysis.

The AI workflow, identified risks, corrections, and verification steps are documented in `ai/prompt_log.md`.

## Reproducibility and verification

Run the following commands from the project root before submission:

```bash
python -m pytest -q
python scripts/check_handin.py
```

The submitted version should pass all automated tests and contain no `[FAIL]` messages from the hand-in checker. The complete submission process is documented in `SUBMISSION_CHECKLIST.md`.

Before hand-in, confirm that:

- `report/report.pdf` is present;
- the live Streamlit application loads successfully;
- the GitHub repository and Streamlit application are publicly accessible;
- both links work in a signed-out or incognito browser.

## Disclaimer

RiskBridge Funds is an educational prototype based on historical out-of-sample backtests. The reported results are illustrative, do not constitute investment advice, and do not guarantee future performance.
