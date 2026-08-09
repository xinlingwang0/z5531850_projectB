# Data Guide (provided - do not edit)

All data loads through `src/data_access.py`. It downloads one public ZIP once and
caches it, so you never commit data files. To use a local copy offline:

    set FINS_DATA_ZIP=C:\path\to\project_data.zip      # Windows
    export FINS_DATA_ZIP=/path/to/project_data.zip     # macOS / Linux

The default source is the official data ZIP on Google Drive, set in
`src/data_access.py`. If it is down, the helper falls back to a backup host
automatically. There are three files (no sector map; the sector is a column in
equity_prices and news_headlines).

## Datasets

### equity_prices  (load_equity_prices)
- 50 US large-cap stocks across 10 sectors, daily, 2020-2023 (50,300 rows).
- Columns: ticker, date, open, high, low, close, adjClose, volume, sector.
- Use adjClose for returns. Equities trade about 252 days a year.

### crypto_prices  (load_crypto_prices)
- 10 cryptocurrencies, daily (14,620 rows): BTC-USD, ETH-USD, ADA-USD, BCH-USD, EOS-USD, ETC-USD, LTC-USD, TRX-USD, XLM-USD, XRP-USD (note the `-USD` suffix).
- Same columns as equities but NO sector column - crypto is price-only.
- Spans 2020-01-01 to 2024-01-01: there are 10 stray rows dated 2024-01-01, so cap your sample at 2023-12-31.
- Crypto trades 7 days a week. Compute returns within each panel, then left-merge crypto onto the equity trading calendar to build the combined panel (this drops weekend-only crypto moves, which is intended).
- Correct: compute each coin's daily returns on its own calendar, then reindex/left-join those returns to equity trading dates. Incorrect: merge equity and crypto price levels first, then difference (that creates spurious returns).

### news_headlines  (load_news_headlines)
- Daily news HEADLINES for the 50 equities, 2020-2023. Headlines only, no body text.
- Columns: date, ticker, sector, title, url, publisher (publisher is often blank).
- 149,683 rows before de-duplication: about 2,847 are exact duplicates (ticker+date+title), so dedup before counting.
- Many rows per ticker-date is normal, so a duplicate check on ticker-date alone flags everything; check exact duplicates on ticker+date+title instead.
- The `date` is timezone-aware UTC (`datetime64[us, UTC]`) while the price dates are tz-naive (`datetime64[ns]`). Normalise timezone and dtype before merging, or `merge_asof` will error.
- Align every headline to its equity trading day (the same day if it is a trading day, otherwise the next trading day), not only weekend ones.
- Headline sentiment is a noisy proxy. Lag it when you trade (a Station 3 concern).

## The 10 sectors (5 stocks each)

- Tech: NVDA, AMD, INTC, QCOM, ADBE
- Financials: GS, MS, WFC, V, USB
- Energy: XOM, CVX, COP, SLB, OXY
- Consumer: DIS, WMT, NKE, SBUX, KO
- Industrials: GE, BA, CAT, UPS, MMM
- Healthcare: MRK, ABBV, AMGN, GILD, ABT
- Comm/Telecom: T, CMCSA, TMUS, EA, TTWO
- Materials: SHW, NEM, DOW, NUE, DD
- Utilities: NEE, DUK, SO, D, AEP
- Real Estate: AMT, O, PLD, CCI, PSA

## Known traps (read before you start)

- Look-ahead bias: form weights and sentiment signals only from past data.
- Calendar mismatch: annualise equities with sqrt(252), crypto with sqrt(365).
- Headlines, not articles: a headline can be neutral while the news is not.
- Neutral sentiment: about half of finance headlines score neutral with plain
  VADER, and many are false neutrals. A finance lexicon helps.
- Thin sectors: Materials, Utilities, and Real Estate have sparser news.
- Coverage cap: the data ends in 2023; do not claim results past 2023-12-31.
- First load: the helper downloads ~11 MB from Google Drive and caches it. The first
  call may be slow or fall back to the backup host automatically - this is normal.
- VADER setup (Part B): nltk's VADER needs a one-time `nltk.download('vader_lexicon')`
  before it scores, or `SentimentIntensityAnalyzer()` raises a LookupError.
- Streamlit cache warnings: running the provided scripts outside the app may print
  `No runtime found, using MemoryCacheStorageManager` warnings, because the data
  helper's cache is built for Streamlit. They are harmless - if the script finishes
  and writes your results/ files, ignore them.
