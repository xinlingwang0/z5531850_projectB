# Project Context (provided - do not edit)

This file grounds your AI assistant in what the project is. For the full rules and
the marking rubric, read PROJECT_BRIEF.md in this folder.

## The product

You are building your own prototype FinTech app (you give it a name): an
investment product offering several systematically managed funds. An investor
compares the funds, reads each fund's fact sheet, and allocates money across them.
The funds and their out-of-sample backtested fact sheets are the product.

## The two Parts

- Part A (Data Foundation, Stations 1-2): load and clean the data, run integrity
  and outlier checks, build return features, and assemble the headlines into a
  daily text panel. The sentiment model is NOT built here.
- Part B (Funds, Sentiment & App, Stations 3-4): the out-of-sample portfolio
  optimisation, the sentiment model that scores the headlines into a sector
  sentiment index, a sentiment-fusion extension, and a deployed Streamlit app.

## The Data Factory Floor (DFF)

Four stations, one per stage: (1) Data Lake / ETL, (2) Feature Engineering,
(3) Model Design, (4) Implementation. Part A = Stations 1-2, Part B = Stations 3-4.
The sentiment model and the portfolio optimisation both live in Station 3 (Part B).

## What the reports must SHOW (self-contained tables and figures)

The canonical list of required exhibits is in PROJECT_BRIEF.md - Section 4 (Part A)
and Section 5 (Part B). Build exactly those, each captioned, labelled (axes, units,
sample period), and interpreted in the text. Note: Part A has NO sentiment-index
figure - scoring and indexing sentiment is Part B.

## Required minimum (funds, Part B)

At least a combined equity-plus-crypto fund with two optimisation methods,
backtested out-of-sample with no look-ahead. Equity-only and crypto-only funds and
extra methods lift the higher bands.

## Data

Load everything through src/data_access.py. See context/DATA_GUIDE.md. Never commit
data files.
