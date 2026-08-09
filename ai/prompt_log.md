# AI Workflow Log - FINS3645 Part B

## Prompt Log 1 - Portfolio construction and walk-forward backtesting

### What I wanted

I wanted a reproducible src/portfolios.py implementation for Project B that constructs long-only Equal Weight, Minimum Variance, 
Maximum Sharpe, and HRP funds and evaluates them with a monthly walk-forward out-of-sample backtest. The implementation needed lagged
portfolio formation, buy-and-hold weight drift, transaction costs, fallback diagnostics, complete holdings records, and fact-sheet performance measures.

### Prompt(s)

"Design and review src/portfolios.py for equity-only, crypto-only, and combined funds. Check the full formation-date → effective-date → holding-return sequence 
for look-ahead bias; verify long-only, fully invested weights; implement monthly out-of-sample rebalancing, transaction costs, fallback behaviour, growth of $1, 
annualised return and volatility, Sharpe ratio, and maximum drawdown. Test adjusted-close returns, crypto calendar alignment, portfolio-weight constraints, 
known performance sequences, optimisation failure, and whether different methods produce genuinely different weights."

### What the assistant produced

The assistant produced a portfolio engine with:

four construction methods: get_equal_weights(), get_min_variance_weights(), get_tangency_weights(), and get_hrp_weights();

_build_target_schedule() and run_walk_forward_backtest() for monthly formation, next-observation implementation, weight drift, turnover, and costs;

SLSQP optimisation plus Equal Weight fallback and rebalance diagnostics;

calculate_growth_of_1(), drawdown, annualised metrics, and current-holdings extraction; and

separate target-weight, return, and diagnostic panels suitable for saving under results/.

### What was wrong or risky

1. Look-ahead bias in _eligible_assets(): the first draft used returns_df.loc[effective_date] to decide eligibility at formation date (T). 
Changing only an asset's (T+1) return from finite to NaN changed its weight formed at (T), so the backtest could avoid future missing observations or suspensions.

2. Invalid SLSQP solutions could appear valid: _validate_solver_weights() normalised result.x before testing the raw sum and bounds. 
A mocked success=True, x=[2, 2] was silently converted to [0.5, 0.5], masking constraint failure and preventing fallback.

3. alse Maximum-Sharpe convergence: when covariance was zero, get_tangency_weights() returned a constant penalty objective, 
but SLSQP could still report success. Its one-asset shortcut also returned 100% before validating volatility.

4. Scaled-covariance validation bypass: the optimiser correctly needed numerical scaling, but the same scaled covariance was 
used to test variance <= RISK_EPSILON. A portfolio with raw variance below 1e-12 could therefore pass after rescaling, even though its Sharpe denominator was economically undefined.

5. Silent data-type conversion: _validate_returns_frame() used pd.to_numeric(..., errors="coerce"); a corrupted value such as "oops" 
became NaN and could quietly remove an asset. Ticker normalisation could also turn "A" and " A" into duplicate columns without rejecting them.

6. Missing-return masking: holding-period returns were filled with zero. Later, the missing-data check defined an active holding as weight > WEIGHT_TOLERANCE, 
so a real but small weight such as 1e-7 could still encounter NaN or Inf without raising an error.

7. Input and audit weaknesses: the early version assumed a DatetimeIndex when computing drawdown, accepted unsafe transaction-cost values, did not reject bankrupting net returns, 
and omitted zero target weights from the holdings panel. These issues could cause index errors, invalid wealth paths, or incomplete audit records.

8. Annualisation semantics: fixed 252 scaling was inappropriate as a general interface for crypto. Although multiplying both mean and covariance by 
the same positive constant does not change these optimiser weights, hard-coding it obscured the 252/365 calendar distinction and solver conditioning.

### What I changed and why

1. I treated the AI output as a draft and tested its claims before accepting it.

2. I wrote a counterfactual no-look-ahead test: changing only (T+1) data must not alter weights formed at (T). After this failed, 
I changed _eligible_assets() to use only the estimation sample ending on the formation date. The new target remains effective on 
the next available date, so the formation-day decision cannot consume its holding-period return.

3. I injected deliberately invalid optimiser outputs, including [2, 2], non-finite values, and success=False. I then separated 
raw solver validation from harmless floating-point repair: shape, finiteness, [0,1] bounds, and sum-to-one are checked before 
clipping or normalising. Invalid output now raises PortfolioConstructionError and activates the recorded Equal Weight fallback.

4. I separated optimisation scaling from financial validation. _return_moments() scales daily covariance only to improve SLSQP 
conditioning; get_tangency_weights() separately retains raw_cov = clean.cov(ddof=1) and validates w' raw_cov w > RISK_EPSILON 
after both the one-asset and multi-asset branches. Constant-return and near-zero-variance tests now trigger fallback instead of false success.

5. I changed numeric conversion to errors="raise", stripped ticker names and checked duplicates again, rejected infinite returns, 
and retained all universe assets—including zero weights—in every rebalance record. This makes malformed input visible and the holdings panel auditable.

6. I changed _asset_returns_for_day() to classify every strictly positive weight as active. Tests with weight 1e-7 and returns of NaN,
Inf, and None now raise ValueError; only a true zero-weight asset may have its missing return filled with zero.

7. I rewrote drawdown using arrays and an explicit initial wealth of 1, so it works with DatetimeIndex, RangeIndex, or another ordinary index. 
I also required transaction_cost_rate in [0,1), rejected net_return <= -1, and validated annualization_factor as a positive integer.

8. I manually checked a known return path, buy-and-hold drift, first and subsequent turnover, cost timing, weight sums, long-only bounds, 
complete weight panels, fallback diagnostics, and method differentiation. The final regression run passed the previously failing counterexamples, 
so I accepted the revised implementation rather than the assistant's initial draft.


## Prompt Log 2 - Portfolio backtest outputs and visualisation revision

### What I wanted

I wanted the AI assistant to run the Part B portfolio workflow and produce the final evidence needed for the report. This included Equity-Only,
Crypto-Only, and Combined fund results for Equal Weight, Minimum Variance, Maximum Sharpe, and HRP, using a monthly walk-forward out-of-sample
backtest with no look-ahead bias and correct handling of the different equity and cryptocurrency trading calendars.

### Prompt(s)

"Run the portfolio component of Part B. Construct Equity-Only, Crypto-Only and Combined funds and compare Equal Weight, Minimum Variance,
Maximum Sharpe and HRP. Use a monthly walk-forward out-of-sample backtest. At each formation date, use only information available at that time
and apply the resulting weights only to subsequent holding-period returns. Correctly handle the equity and cryptocurrency trading calendars.
Export fund returns, portfolio weights, performance metrics, current holdings and backtest diagnostics. Produce report-ready figures covering
growth of $1, drawdown, asset weights, Sharpe ratios, turnover and solver fallback usage. Each figure should include a clear title, labelled axes,
units, a legend where required, the sample period and a data-source note."

After reviewing the first outputs, I gave follow-up instructions to investigate the all-zero solver fallback result, revise the growth-of-$1 figure,
and replace the Sharpe-ratio scatter plot with a grouped bar chart.

### What the assistant produced

The assistant generated fund return, portfolio weight, performance metric, current-holding, and backtest diagnostic outputs for all three fund
families and four portfolio methods. It also produced report figures for cumulative growth, drawdown, Combined-fund asset-class weights, Sharpe
ratios, turnover, and solver fallback usage.

The main calculations were usable, but several visual and diagnostic outputs needed further checking before I could rely on them in the report.

### What was wrong or risky

1. The solver fallback figure showed a 0% fallback rate for every fund family and method. This could have been a genuine result, but it could
also have meant that `fallback_used` was not recorded correctly, Boolean values were converted incorrectly, or the plotting aggregation failed.

2. The first growth-of-$1 figure did not label each method's terminal value. This made it difficult to compare final outcomes, especially where
several portfolio methods ended close together.

3. The growth-of-$1 panels used different y-axis ranges, so the $1 break-even line appeared at different heights across fund families. This could
make cross-panel comparisons look more meaningful than they really were.

4. The first Sharpe-ratio figure used a scatter plot even though both fund family and portfolio method are discrete categories. The values could
still be correct, but the chart type made the comparison harder to read.

### What I changed and why

I treated the AI's first output as a draft rather than accepting it automatically. For the solver fallback result, I asked the assistant to inspect
the raw rebalance records, aggregate fallback usage across all fund-family and method combinations, and cross-check the result against solver
success and diagnostic messages. The follow-up check found 432 rebalance records, all with `fallback_used=False`, and all solver records indicated
success. I therefore kept the 0% fallback result, but revised the figure so it clearly states that the zero fallback rate was verified rather than
leaving the panel looking empty.

For the growth-of-$1 figure, I asked the assistant to add endpoint markers and terminal-value labels for each method, adjust nearby labels to avoid
overlap, and keep a visible $1 break-even reference. I also retained a note explaining that the three panels use independent y-axis scales, so the
main comparison should be within each fund family rather than across panel heights.

For the Sharpe-ratio comparison, I replaced the scatter plot with a grouped bar chart. This better matches the categorical structure of the data:
fund family appears on the x-axis, the four methods are shown as adjacent bars, and each bar is labelled with its Sharpe ratio.

### How I verified it

I verified that the all-zero fallback result came from the underlying diagnostics rather than missing data or a plotting bug. I also checked that
the revised figures communicated the same return and performance data more clearly: growth-of-$1 now reports terminal values, the fallback panel
explicitly documents the verified 0% result, and the Sharpe-ratio chart supports direct comparison across methods and fund families.


## Prompt Log 3 - FinVADER baseline and 2020 candidate discovery

### What I wanted

I wanted a reproducible FinVADER baseline that scores each deduplicated raw headline unchanged, plus a separate candidate-term discovery process
using only headlines published during 2020. The 2021-2023 application period had to remain excluded from candidate frequency, examples, ticker coverage, and sector coverage.

### Prompt(s)

"Implement the current sentiment baseline and 2020 candidate discovery without using features.top_terms(). Preserve raw_title exactly for official 
FinVADER compound scoring with SentiBignomics and Henry enabled. Add a --sentiment-candidates-only mode to scripts/run_part_b.py, keep the default
command working, test the original-date split and unchanged text input, and do not modify portfolio, fusion, dependency, app, or report files."

### What the assistant produced

The assistant added guarded offline loading of FinVADER's public scorer, per-headline scoring, ticker-day aggregation, an equal-weight sector-day index, 
coverage summaries, independent regex candidate discovery, the candidate-only runner mode, and offline tests for text preservation and period isolation.

### What was wrong or risky

The first candidate-only run could not read the hosted course bundle because the restricted process could not resolve the data hosts. No fallback dataset 
was substituted. A rerun through the required src/data_access.py path with course-data access completed successfully. The first unit-test run also exposed 
a display-only path assumption when the output directory was replaced by pytest's temporary directory.

### What I changed and why

I kept scoring and discovery separate: raw_title goes directly to the public FinVADER function, while candidate_terms() tokenises only a 2020 copy for counting.
I made candidate output-path display work both inside and outside the project root, retained stop words, and used the original publication date rather than trading_date 
for the discovery boundary. The effective baseline lexicon follows FinVADER 1.0.2's SentiBignomics scaling and Henry override order only to flag already-covered candidates; 
candidate statistics never alter scoring.

### How I verified it

The offline scorer received a title containing "not", "VERY", and punctuation without any change. The sentiment tests passed after the path fix. The full candidate run produced
results/tables/sentiment_candidate_terms.csv with 17,681 rows from 36,955 headlines dated 2020-01-01 through 2020-12-31; 2021-2023 contributed zero rows to discovery.


## Prompt Log 4 - AI-assisted sentiment lexicon rating and freezing

### What I wanted

I wanted to extend the FinVADER baseline with terms that could be matched directly when scoring financial news. I extracted candidates using only
the 2020 discovery-period headlines and applied prespecified screening requirements, including absence from the existing baseline lexicon, an exact
VADER-token match, a frequency of at least 20, and coverage across at least five sectors. This screening produced 62 candidate terms.

I manually reviewed the 62 candidates and excluded 32 named entities, topic terms, and words whose meanings or sentiment directions were unclear
without context. The remaining 30 terms proceeded to the AI-rating stage. My goal was to estimate their sentiment valence using repeated ratings
from several models and freeze the expanded lexicon after checking rating agreement.

### Prompt(s)

I sent the same standardised prompt to ChatGPT, Gemini, Claude, Grok, and DeepSeek. Each model completed the rating task twice, giving ten ratings
for every term. Only the `RATER_ID` changed between runs; the instructions, term spellings, and term order remained fixed.

The main prompt was:

> You are one of several raters assisting with the construction of an English financial-news sentiment lexicon. You cannot see other raters’ responses and must not assume the scores they assigned.
>
> For each supplied term, rate its usual sentiment valence in financial news, securities-market reporting, and corporate disclosures. Scores must be integers from −4 to +4. A score of −4 represents extremely negative sentiment, 0 represents neutrality, unstable direction, or insufficient information from the isolated term, and +4 represents extremely positive sentiment.
>
> Rate the term itself without using a specific sentence. Use its common meaning in financial news, securities-market reporting, and corporate disclosures. Select 0 when the direction depends on the object or context. Do not convert market co-occurrence into sentiment belonging to the word itself.
>
> Do not rewrite, merge, stem, or omit any term. Report `confidence` from 1 to 5. Set `ambiguous=true` when the term may have opposing directions or normally requires an object or context. Provide a short rationale without inventing sentences, data, research, or other raters’ opinions.
>
> Return only a valid JSON array. Each object must use:
>
> `{"rater_id":"RATER_ID","term":"original term","valence":0,"confidence":1,"ambiguous":true,"rationale":"English rationale of no more than 20 words"}`

The 30 rated terms were:

`jumps`, `jumped`, `selloff`, `sell-off`, `soars`, `soaring`, `surged`, `crashes`, `rallies`, `soared`, `delays`, `downgraded`,
`better-than-expected`, `sinks`, `rebounding`, `climbs`, `upgraded`, `tumbles`, `plunges`, `comeback`, `layoffs`, `rebounds`, `plunged`,
`rallied`, `rout`, `hampered`, `crashed`, `retreats`, `bearish`, and `breakout`.

### What the assistant produced

The five models each returned two structured JSON outputs, giving ten integer valence ratings for every term. Each record also contained a
confidence score, an ambiguity flag, and a rationale of no more than 20 English words.

I combined the ten outputs in an adjudication table. For each term, the table recorded:

- the valence from every run;
- mean valence across the ten ratings;
- sample standard deviation;
- `n_ratings`;
- `ambiguous_count`;
- whether the agreement rule passed;
- the final inclusion decision; and
- frozen valence.

Final valence was the arithmetic mean of the ten integer ratings, rounded to one decimal place. All 30 terms satisfied the prespecified
`sample SD ≤ 2.5` agreement rule and were included in the expanded lexicon. The final frozen lexicon retained only the original `term` and its
corresponding `valence`.

### How I verified it

I confirmed that the manually screened list contained 30 terms and that every term received ten ratings. All valence scores were integers between
−4 and +4. No terms were omitted, rewritten, merged, or stemmed.

I checked the mean valence, sample standard deviation, `n_ratings`, and final decision recorded for every term in the adjudication table. Every
term had ten ratings, every sample SD was no greater than 2.5, and every final decision was recorded as `include`.

Finally, I compared the frozen lexicon with the adjudication table. The frozen file contained the same 30 original terms, and each one-decimal
frozen valence matched the corresponding final value in the adjudication table.


## Prompt Log 5 - Sentiment implementation and final review

### What I wanted

I wanted the AI to complete the RiskBridge Funds Part B sentiment workflow by integrating the frozen 30-term lexicon into FinVADER and building
comparable Baseline and Extended models.

The workflow had to produce 2021-2023 sector sentiment indices, comparison tables, Figure 6, and Figure 7. Aggregation had to follow headline to
ticker-day to sector-day, with no-news observations kept as `NaN` and standardisation based only on prior history. Fusion, portfolio, and app work
were excluded.

### Prompt(s)

I used three rounds of prompts:

1. **Initial implementation:** build the Baseline and Extended models, sector index, comparison outputs, figures, and tests without changing the
   frozen lexicon.
2. **First review:** fix the package import, align the Overall definition used by the comparison table and Figure 7, and improve Figure 6's
   explanation.
3. **Final review:** stop Figure 6's rolling line from covering missing dates and set Matplotlib to the `Agg` backend before importing plotting
   code.

### What the assistant produced

The AI produced:

- Baseline and Extended headline scores;
- ticker-day and sector-day aggregates;
- expanding z-scores and five-band classifications;
- `results/data/sector_sentiment_index.csv` and model-comparison tables;
- Figures 6 and 7; and
- sentiment unit tests and output checks.

### What was wrong or risky

The review identified several issues:

- `import sentiment_lexicon` depended on manual `sys.path` changes;
- the comparison table and Figure 7 used different Overall definitions;
- Figure 6 did not clearly explain its z-scores and blank segments;
- the rolling line covered 137 missing daily sector observations; and
- the Figure 7 test could load the macOS GUI backend and crash in a headless environment.

### What I changed and why

I required a proper relative import:

```python
from . import sentiment_lexicon
```

Overall sentiment was redefined as the daily equal-weight mean across sectors with valid observations. The same aggregate was then used for both
the comparison table and Figure 7.

Figure 6 retained its 5 by 2 design, but its labels and notes were clarified. Its rolling mean now uses the latest 21 valid observations and is
reindexed to the full date series, so it remains `NaN` whenever the daily index is missing.

The test file also sets the backend before importing `run_part_b`:

```python
import matplotlib

matplotlib.use("Agg")
```

### How I verified it

I checked that:

- `src.sentiment` imports correctly from the project root;
- Figure 7 and the comparison table share the same Overall aggregate;
- rolling coverage of missing dates fell from 137 cases to zero;
- Figures 6 and 7 contain only 2021-2023 data;
- sentiment tests run without manually setting the `MPLBACKEND` environment variable; and
- the frozen lexicon, scoring, aggregation, no-news treatment, expanding standardisation, fusion, portfolio, and app code were unchanged during
  the implementation and review rounds.



## Prompt Log 6 - Fusion and sentiment tilt

### What I wanted

I asked the AI to complete the Part B fusion module by applying the Extended sentiment signal to the Equity-Only Minimum Variance portfolio and
comparing three predefined variants:

- Base fund (`lambda = 0`)
- Momentum tilt (`lambda = +1`)
- Contrarian tilt (`lambda = -1`)

All variants had to use the same backtesting engine and avoid look-ahead bias.

### Prompt(s)

My main instruction was:

> Extract a shared target-schedule backtesting function from the existing portfolio workflow and confirm that the original 12 funds remain
> unchanged after refactoring. Use only the Extended sentiment signal for fusion and do not rerun the Baseline model. Apply lagged ticker-level
> sentiment signals at monthly rebalancing dates, preserve missing values, and produce the three fusion variants, performance outputs, coverage
> diagnostics, Figure 8, and automated tests. Do not modify sentiment, the app, or the report.

### What the assistant produced

The AI produced:

- a shared target-schedule backtesting interface;
- ticker-level Extended expanding z-scores;
- Base, Momentum, and Contrarian variants;
- returns, weights, turnover, transaction costs, and performance outputs;
- sentiment coverage diagnostics and Figure 8; and
- tests for signal timing, missing values, weight constraints, and look-ahead bias.

### What was wrong or risky

I found no material errors or additional risks requiring correction. The implementation followed the requested scope and preserved the existing
portfolio and sentiment logic.

### What I changed and why

I made no additional changes because the generated implementation met the original requirements.

### How I verified it

I checked the pre- and post-refactor portfolio results, signal lags, weight validity, missing-signal treatment, and automated test results. I also
confirmed that the sentiment module, app, report, and existing fund results were not modified.


## Prompt Log 7 - Streamlit app fixes and testing

### What I wanted

I wanted the AI to make a small, controlled improvement to the Streamlit app without changing the models, data outputs, or existing page structure.
Only `streamlit_app.py` and `tests/test_app.py` could be modified during the implementation and testing task.

### Prompt(s)

I asked the AI to:

- fix stale allocation weights after changing the selected funds;
- preserve manual weights when the fund selection remains unchanged;
- keep percentages, weights, and Sharpe ratios numeric in financial tables;
- change the Growth of $1 chart from a logarithmic to a linear axis;
- improve the Equal Weight explanation and Fusion turnover label;
- add AppTests and run the required validation checks; and
- confirm that all protected files remained unchanged.

### What the assistant produced

The AI revised the allocation session-state logic so that a changed fund selection receives approximately equal weights totalling exactly 100%,
while manual inputs remain unchanged during normal reruns.

It also retained numeric data types through Streamlit column configuration, changed Growth of $1 to a linear scale, and added tests for allocation
interactions, table data types, and chart behaviour.

### What was wrong or risky

The original app had three main issues:

- switching from three funds to two or four could leave stale widget values, producing totals of 66.66% or 125%;
- several financial values were converted to strings, causing text-based rather than numeric sorting; and
- Growth of $1 used an unexplained logarithmic axis.

### What I changed and why

I required the app to clear obsolete allocation state and reset weights only when the selected funds changed. User-entered weights had to remain
intact when the selection stayed the same.

Financial columns were kept numeric and formatted with `NumberColumn`. Growth of $1 was changed to a linear axis for consistency with the other
growth charts.

### How I verified it

I checked that:

- weights still totalled 100% after switching from three funds to two or four;
- manual weights persisted when the selection was unchanged;
- removed and re-added funds did not recover stale weights;
- key financial columns remained numeric;
- the Growth of $1 axis was not `log` and its underlying data was unchanged; and
- only `streamlit_app.py` and `tests/test_app.py` were modified during the implementation and testing task.


## Prompt Log 8 - Adding and correcting sentiment figures

### What I wanted

I wanted the AI to add two figures to strengthen the RiskBridge Funds Part B sentiment analysis:

- Figure 9: market-wide news sentiment over 2021-2023; and
- Figure 10: average sentiment rankings across equity sectors.

Both figures had to use the existing Extended FinVADER sector index without rescoring headlines or changing the sentiment, fusion, portfolio, app,
or report outputs.

### Prompt(s)

My initial prompt asked the AI to:

- build Figure 9 from the shared daily Overall aggregate;
- show the market sentiment level, a 21-valid-observation rolling mean, and relative historical sentiment deviations;
- rank sectors in Figure 10 using mean Extended compound scores;
- preserve missing values and avoid weighting by news volume;
- integrate both figures into the existing build; and
- add automated tests and conduct visual checks.

After reviewing the output, I requested four corrections:

- prevent Figure 9 from displaying 2020 or 2024 on its axis;
- correct Figure 10's axis label because it displays both Mean and Median;
- replace the inaccurate "baseline vs Extended" source note with an Extended-only note; and
- make missing required outputs fail tests instead of triggering `pytest.skip`.

### What the assistant produced

The AI added Figures 9 and 10 to the sentiment build. Figure 9 presents the market-wide sentiment level and average sector z-score, while Figure 10
ranks all ten sectors using their mean, median, and valid observation counts.

It also added tests for date limits, aggregation, rolling missing values, sector rankings, plotted elements, and required output files.

### What was wrong or risky

My review identified four issues:

- Figure 9's automatic axis margins displayed dates from 2020 and 2024;
- Figure 10's axis label did not accurately describe both Mean and Median;
- the source note incorrectly implied a Baseline comparison; and
- skipped missing-output tests could conceal a failure to generate the figures.

### What I changed and why

I required Figure 9's axis to be restricted to the actual data period and changed Figure 10's axis label to:

`Extended FinVADER compound score`

Both figures were given an Extended-only source note identifying the 30-term custom lexicon and the 2021-2023 application period. Tests were also
changed to fail with clear messages when required inputs or PNG files were missing.

### How I verified it

I checked that:

- Figure 9's data and visible ticks were limited to 2021-2023;
- Figure 10's Mean, Median, ranking, and axis label were correct;
- neither source note contained "baseline vs Extended";
- missing required inputs or PNGs caused test failures rather than skips;
- the hashes of Figures 1-8 remained unchanged; and
- no sentiment calculations, fusion, portfolio, app, report, or existing CSV outputs were modified.


## Prompt Log 9 - Private GitHub deployment preparation

### What I wanted

I wanted the AI to prepare the completed RiskBridge Funds Part B project for deployment without changing any analysis, results, tests, or
Streamlit code.

### Prompt(s)

My instruction was:

> Confirm that the working directory is `z5531850_projectB`, read `AGENTS.md` and `docs/STUDENT_DEPLOY.md`, and run
> `scripts/check_handin.py`. Do not modify `src/`, `streamlit_app.py`, `tests/`, or `results/`. Initialise Git in the project folder,
> commit the code and generated results, and push them to a new private GitHub repository. Streamlit deployment itself remains my
> responsibility because it requires my login.

### What the assistant produced

The AI completed the deployment-preparation workflow by checking the project location and instructions, validating the hand-in structure,
initialising the repository, committing the required project files, and pushing them to a new private GitHub repository.

### What was wrong or risky

I found no material errors or issues requiring correction. The task remained limited to repository and deployment preparation.

### What I changed and why

I made no follow-up changes because the AI followed the requested scope and did not alter the completed analysis or application.
