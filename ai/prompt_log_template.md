# Prompt log - <task name>

## What I wanted
<the goal in one or two sentences>

## Prompt(s)
<the prompt(s) you gave the assistant>

## What the assistant produced
<a summary, or the code/text it returned>

## What was wrong or risky
<bugs, look-ahead, wrong assumptions, hallucinated APIs - what you found>

## What I changed and why
<your correction, in your own words>

---

## Example (one filled-in entry - delete before you hand in)

### What I wanted
A function to compute simple daily returns per ticker from the equity panel.

### Prompt(s)
"Write a function that pivots equity_prices to wide adjClose and returns daily simple
returns per ticker."

### What the assistant produced
A `daily_returns()` that pivoted on `close` (not `adjClose`) and used `.diff()`
instead of `.pct_change()`.

### What was wrong or risky
Two bugs: it used raw close (ignores splits and dividends) and `.diff()` gives price
changes, not returns.

### What I changed and why
Switched to `adjClose` and `.pct_change()`, and confirmed the first row is NaN per
ticker. Checked one value by hand for AAPL on a single date.
