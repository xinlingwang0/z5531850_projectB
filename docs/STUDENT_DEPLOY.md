# Student Guide: Deploying Your Streamlit App for Hand-in

The path from a local app to the public URL you submit for Part B. Your
`<zID>_projectB` folder is its own GitHub repository, and the app entrypoint is
`streamlit_app.py` at the folder root. The repo stays **private while you work** and
becomes **public at hand-in** so markers can run it.

**Who does what:** your AI agent can build the app, run `scripts/check_handin.py`,
and push the repo. The final deploy on Streamlit Community Cloud is **browser-based
and needs your own GitHub and Streamlit login**, so you do that step yourself.

## What you submit (Part B, Station 4)
- A **live public Streamlit Community Cloud URL**.
- A **public GitHub repository** (the contents of your `<zID>_projectB` folder).
- The repo runs from a clean checkout: raw data loads from the hosted ZIP through
  `src/data_access.py`, and your precomputed app artifacts are committed under
  `results/`.

## Step-by-step

1. **Start from your folder.** You already have `<zID>_projectB` (unzipped from
   `projectB_starter.zip` and renamed). It contains `streamlit_app.py`,
   `.streamlit/config.toml`, `requirements.txt`, and `src/data_access.py`.
2. **Build your app.** Replace the starter `streamlit_app.py` with your real
   dashboard: a fund picker, each fund's fact sheet (growth of $1, drawdown, Sharpe,
   holdings), an allocation control, and your sentiment analytics. Load raw data
   through `src/data_access.py`; do not hard-code laptop paths.
3. **Keep the app light.** Put modelling in functions and cache with `st.cache_data`.
   The app must run on the free tier (about one core), so **precompute the slow
   backtests and the sentiment index, save them under `results/data/`, and have the
   app load those** rather than recompute. Keep `nltk` out of the app (it lives in
   `requirements-dev.txt`).
4. **Run and check locally:**
   - `streamlit run streamlit_app.py`
   - `python scripts/check_handin.py` (naming, entrypoint, requirements, no raw data
     or secrets committed).
5. **Make it a GitHub repo.** Inside `<zID>_projectB`: `git init`, commit your code
   AND your `results/` artifacts (the deployed app reads them), and push to a **new**
   repo. Keep it **private** while you build. The folder is the repo root, so the
   Streamlit entrypoint is just `streamlit_app.py`.
6. **You deploy in the browser.** On share.streamlit.io, sign in, New app -> pick
   your repo, branch `main` (or `master`, whichever holds the app), main file
   `streamlit_app.py`, Python 3.13. Your AI agent cannot do this step - it needs your
   login. The app can run from a private repo while you develop.
7. **At hand-in: make the repo PUBLIC**, confirm the live URL loads in a fresh
   incognito browser (logged out), and submit the public URL + the repo URL, plus the
   zipped folder to Moodle.

## Nested-repo caveat
If you keep your project inside `fins-agent/fins2026/`, do **not** push it as part of
the fins-agent repository. `git init` inside your `<zID>_projectB` folder and push
that folder's contents to a **separate, fresh** GitHub repo. (Tip: add
`fins2026/z*_project*` to the fins-agent `.gitignore` so it stays out of that repo.)

## Common pitfalls
- **Missing app artifact.** The app errors on Cloud because a precomputed file under
  `results/data/` was not committed. The starter `.gitignore` keeps `results/`
  committed while blocking raw data - do not re-ignore it.
- **Missing requirement.** Test in a fresh virtual environment; the app's deps are in
  `requirements.txt` (no `nltk`).
- **Absolute paths or committed raw data.** These break on Cloud. Load raw data
  through `src/data_access.py`.
- **Private at hand-in.** Markers cannot open a private app. Make it public and
  re-test the URL before the deadline.
- **Heavy compute on every click.** Cache, or precompute and load from `results/`.
- **Cold start.** The first data load downloads the hosted ZIP; cache it.

## Troubleshooting (failures we actually hit)
- **`gh` lost auth mid-session** (push fails): re-run `gh auth login -h github.com -w`
  and push again.
- **Your private repo is not in Streamlit's picker:** link your GitHub account in
  Streamlit settings, or use "Paste GitHub URL" and paste the URL of the
  `streamlit_app.py` file.
- **`results/data` not committed -> the app errors on Cloud:** the starter
  `.gitignore` keeps `results/` committed; commit your precomputed artifacts and push
  before deploying.
- **VADER `LookupError` on a clean machine:** run `nltk.download('vader_lexicon')`
  once in your build (a `run_part_b.py` step, not the app).
