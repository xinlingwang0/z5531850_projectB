"""Pre-hand-in check. Run from the project root:

    python scripts/check_handin.py

Fix every [FAIL] before you zip and submit. [WARN] items are reminders (they do not
block). Part B additionally checks the app and deploy files.
"""
import re
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
oks, problems, warns = [], [], []


def check(cond, ok_msg, bad_msg):
    (oks if cond else problems).append(ok_msg if cond else bad_msg)


def warn(cond_ok, ok_msg, warn_msg):
    (oks if cond_ok else warns).append(ok_msg if cond_ok else warn_msg)


name = ROOT.name
m = re.fullmatch(r"z[0-9]{7}_project([AB])", name)
check(bool(m), f"folder name '{name}' is valid",
      f"folder name '{name}' must be z<7 digits>_projectA or _projectB")
part = m.group(1) if m else None

for rel in ["README.md", "SUBMISSION_CHECKLIST.md", "ai",
            "context/DATA_GUIDE.md", "src/data_access.py", "report"]:
    check((ROOT / rel).exists(), f"{rel} present", f"missing {rel}")

# At least one agent file must be your own (you only use one tool, so both need
# not be edited - but at least one must be).
agent_edited = [f for f in ("AGENTS.md", "CLAUDE.md")
                if (ROOT / f).exists()
                and "replace this placeholder" not in (ROOT / f).read_text(encoding="utf-8", errors="ignore").lower()]
check(bool(agent_edited), f"your own agent file ({', '.join(agent_edited) or 'none'})",
      "edit AGENTS.md or CLAUDE.md (your tool's file) with your own instructions - both are still the provided stub")

# Build the token dynamically so the stamp tool does not rewrite it in this file.
placeholder = "z" + "X" * 7
placeholder_left = [
    str(p.relative_to(ROOT)) for p in ROOT.rglob("*")
    if p.is_file() and p.suffix.lower() in {".md", ".py", ".txt", ".toml"}
    and placeholder in p.read_text(encoding="utf-8", errors="ignore")
]
check(not placeholder_left, "no leftover zID placeholder",
      f"placeholder {placeholder} still in: {placeholder_left[:3]}")

data_files = [str(p.relative_to(ROOT)) for p in ROOT.rglob("*")
              if p.suffix.lower() in {".parquet", ".csv"} and "results" not in p.parts]
check(not data_files, "no committed data files",
      f"data files should not be committed: {data_files[:3]}")

junk = [str(p.relative_to(ROOT)) for p in ROOT.rglob("*")
        if p.is_file() and (".tmp." in p.name or p.name in {".DS_Store", "Thumbs.db"})]
check(not junk, "no stray editor-backup or OS-junk files",
      f"remove these before zipping: {junk[:3]}")

pycache = [str(p.relative_to(ROOT)) for p in ROOT.rglob("*")
           if p.is_file() and (p.suffix == ".pyc" or "__pycache__" in p.parts)]
warn(not pycache, "no compiled-Python clutter",
     "delete __pycache__/ and *.pyc before you zip - they are auto-generated and not needed")

# Reminders (do not block): is the actual work there yet?
report = (ROOT / "report" / "report.pdf").exists() or (ROOT / "report" / "report.docx").exists()
warn(report, "report present (report/report.pdf or .docx)",
     "no report/report.pdf yet - author it in Word and export to PDF")
results_files = [p for p in (ROOT / "results").rglob("*")
                 if p.is_file() and p.name != ".gitkeep"]
warn(bool(results_files), f"results/ has {len(results_files)} output file(s)",
     "results/ has no figures or tables yet - save your exhibits there")

# Required output filenames - markers (and, for Part B, the app) rely on these exact names.
required_outputs = {
    "A": ["results/tables/dataset_inventory.csv",
          "results/tables/descriptive_stats_returns.csv"],
    "B": ["results/data/fund_returns.csv", "results/data/fund_weights.csv",
          "results/data/sector_sentiment_index.csv",
          "results/tables/performance_metrics.csv"],
}.get(part, [])
for rel in required_outputs:
    warn((ROOT / rel).exists(), f"{rel} present",
         f"expected output {rel} not found - use this exact name so markers can find it")

if part == "B":
    for rel in ["streamlit_app.py", "requirements.txt", ".streamlit/config.toml"]:
        check((ROOT / rel).exists(), f"{rel} present", f"missing {rel} (needed to deploy)")
    check(not (ROOT / ".streamlit" / "secrets.toml").exists(),
          "no committed secrets", ".streamlit/secrets.toml must not be committed")
    app_text = ((ROOT / "streamlit_app.py").read_text(encoding="utf-8", errors="ignore")
                if (ROOT / "streamlit_app.py").exists() else "")
    warn("nltk" not in app_text, "app does not import nltk",
         "streamlit_app.py references nltk - the deployed app must read precomputed results/, not run VADER (the free tier cannot)")

print(f"\n{len(oks)} checks passed.")
if warns:
    print(f"{len(warns)} reminder(s):")
    for w in warns:
        print("  [WARN]", w)
if problems:
    print(f"{len(problems)} problem(s) to fix:")
    for p in problems:
        print("  [FAIL]", p)
    sys.exit(1)
print("All checks passed - ready to zip" + (" and deploy." if part == "B" else "."))
