"""
Clean and expand the dataindex.us Federal Data Terminations Tracker.

Inputs  (put them in the same folder, or pass paths as arguments):
  1. the raw tracker CSV
  2. entries_2.csv      -> only used to copy over the two OIRA date columns
  3. agency_lookup.csv  -> agency abbreviation -> full name + parent department

Outputs:
  tracker_clean.csv      one row per entry (375 rows)
  sogi_phrases_long.csv  one row per entry + phrase (data-element entries only)

Usage:
  python clean_tracker.py RAW_CSV ENTRIES_2_CSV [OUTPUT_FOLDER]
"""
import re
import sys
from urllib.parse import urlparse

import numpy as np
import pandas as pd

RAW_CSV = sys.argv[1] if len(sys.argv) > 1 else "federal_terminations_tracker.csv"
DATES_CSV = sys.argv[2] if len(sys.argv) > 2 else "entries_2.csv"
OUT = sys.argv[3] if len(sys.argv) > 3 else "."
LOOKUP_CSV = "agency_lookup.csv"

# Phrases for the pie chart: label -> regex (case-insensitive). Edit freely.
PHRASES = {
    "sex at birth": r"sex (?:assigned )?at birth",
    "transgender": r"transgender",
    "nonbinary": r"non[- ]?binary",          # also catches "non-binary" / "non binary"
    "sexual orientation": r"sexual orientation",
    "gender identity": r"gender identity",
}
TEXT_COLS = ["description", "caveats"]        # where to look for the phrases
DOTS_PER_ROW = 25                             # width of the one-dot-per-entry grid


# ---------------------------------------------------------------- 1. load + add dates
raw = pd.read_csv(RAW_CSV)
dates = pd.read_csv(DATES_CSV, usecols=["dataset_name", "icr_received_date", "icr_submitted_date"])
df = raw.merge(dates, left_on="Dataset Name", right_on="dataset_name", how="left", validate="one_to_one")
df = df.drop(columns="dataset_name")
assert len(df) == len(raw), "merge changed the row count"
assert df.icr_received_date.notna().sum() == dates.icr_received_date.notna().sum(), "some dates did not match"

# ---------------------------------------------------------------- 2. rename columns
df = df.rename(columns={
    "Dataset Name": "dataset_name", "Acronym": "acronym", "Description": "description",
    "Caveats": "caveats", "Agency": "agency", "Data Type": "topic", "Status": "status",
    "Date Terminated": "date_terminated",
    "Link 1 URL": "link_1_url", "Link 1 Text": "link_1_text", "Link 1 Source": "link_1_source",
    "Link 2 URL": "link_2_url", "Link 2 Text": "link_2_text", "Link 2 Source": "link_2_source",
    "Link 3 URL": "link_3_url", "Link 3 Text": "link_3_text", "Link 3 Source": "link_3_source",
    "Primary Link Wayback": "archive_url",
})

# ---------------------------------------------------------------- 3. clean text
def fix_text(s):
    """Trim, collapse whitespace, and repair garbled apostrophes (e.g. 'Veterans‚Äô' -> 'Veterans’')."""
    if not isinstance(s, str):
        return s
    s = " ".join(s.split())
    try:
        s = s.encode("mac_roman").decode("utf-8")   # only succeeds on genuinely garbled text
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return s or pd.NA

text_cols = df.select_dtypes(include=["object", "string"]).columns
before = df[text_cols].copy()
df[text_cols] = df[text_cols].apply(lambda col: col.map(fix_text))
print("Text cells repaired/trimmed:", int((before.fillna("") != df[text_cols].fillna("")).sum().sum()))

# ---------------------------------------------------------------- 4. level, IDs, agency names
df["level"] = df["Section"].map({"Data Elements Removed": "Data element", "Datasets Terminated": "Dataset"})
assert df.level.notna().all(), "unexpected Section value"
df = df.drop(columns="Section")
df.insert(0, "entry_id", [f"DI-{i + 1:03d}" for i in range(len(df))])   # follows the tracker's row order

lookup = pd.read_csv(LOOKUP_CSV)
df = df.merge(lookup, on="agency", how="left", validate="many_to_one")
assert df.agency_name.notna().all(), f"agencies missing from lookup: {set(df.agency[df.agency_name.isna()])}"

# ---------------------------------------------------------------- 5. dates
for c in ["date_terminated", "icr_received_date", "icr_submitted_date"]:
    df[c] = pd.to_datetime(df[c])

# ICR reference number (e.g. 202502-0970-008) is found inside the Reginfo.gov links.
# If a row cites several ICRs, the first one is used.
link_text = df[["link_1_url", "link_2_url", "link_3_url"]].fillna("").agg(" ".join, axis=1)
df["icr_ref_nbr"] = link_text.str.extract(r"ref_nbr=(\d{6}-\d{4}-\d{3})")[0]

# One date for timelines: date received by OIRA, else date submitted to OIRA.
df["filing_date"] = df.icr_received_date.fillna(df.icr_submitted_date)
df["filing_date_type"] = np.select(
    [df.icr_received_date.notna(), df.icr_submitted_date.notna()],
    ["Received by OIRA", "Submitted to OIRA"], default=None)
df["filing_month"] = df.filing_date.dt.to_period("M").dt.to_timestamp()

# Sanity check: the date should fall close to the month in the ICR number.
ref_month = pd.to_datetime(df.icr_ref_nbr.str[:6], format="%Y%m")
months_after = (df.filing_date.dt.year - ref_month.dt.year) * 12 + (df.filing_date.dt.month - ref_month.dt.month)
df["filing_date_review"] = (months_after > 6) | (months_after < 0)   # True = double-check this date

# ---------------------------------------------------------------- 6. link evidence (for the dot chart)
link_cols = ["link_1_url", "link_2_url", "link_3_url", "archive_url"]
hosts = df[link_cols].apply(lambda col: col.map(lambda u: urlparse(u).netloc.lower() if isinstance(u, str) else ""))
df["has_archive_link"] = hosts.apply(lambda c: c.str.endswith(("archive.org", "archive.ph", "archive.is"))).any(axis=1)
df["has_government_link"] = hosts.apply(lambda c: c.str.endswith((".gov", ".mil"))).any(axis=1)
df["evidence_class"] = np.select(
    [df.has_government_link & df.has_archive_link, df.has_government_link, df.has_archive_link],
    ["Government and archive link", "Government link only", "Archive link only"], default="Neither")

# Grid position for a one-dot-per-entry chart (top-left = first entry; no axis flip needed).
i = np.arange(len(df))
n_rows = int(np.ceil(len(df) / DOTS_PER_ROW))
df["dot_x"] = i % DOTS_PER_ROW + 1
df["dot_y"] = n_rows - i // DOTS_PER_ROW

# ---------------------------------------------------------------- 7. SOGI phrases (for the pie chart)
text = df[TEXT_COLS].fillna("").agg(" ".join, axis=1)
counts = pd.DataFrame({name: text.str.count(pattern, flags=re.IGNORECASE) for name, pattern in PHRASES.items()})
df["mentions_sogi_phrase"] = counts.sum(axis=1) > 0

long = (counts[df.level == "Data element"]
        .join(df[["entry_id", "agency"]])
        .melt(id_vars=["entry_id", "agency"], var_name="phrase", value_name="mention_count")
        .query("mention_count > 0")
        .sort_values(["phrase", "entry_id"]))

# ---------------------------------------------------------------- 8. save
columns = ["entry_id", "dataset_name", "acronym", "description", "caveats",
           "agency", "agency_name", "parent_department", "topic", "level", "status",
           "date_terminated", "icr_ref_nbr", "icr_received_date", "icr_submitted_date",
           "filing_date", "filing_month", "filing_date_type", "filing_date_review",
           "link_1_url", "link_1_text", "link_1_source", "link_2_url", "link_2_text", "link_2_source",
           "link_3_url", "link_3_text", "link_3_source", "archive_url",
           "has_government_link", "has_archive_link", "evidence_class", "mentions_sogi_phrase",
           "dot_x", "dot_y"]
df[columns].to_csv(f"{OUT}/tracker_clean.csv", index=False)
long.to_csv(f"{OUT}/sogi_phrases_long.csv", index=False)

# ---------------------------------------------------------------- 9. quick checks
assert df.entry_id.is_unique and len(df) == 375
print(f"\nRows: {len(df)} | with filing date: {df.filing_date.notna().sum()} | dates to double-check: {df.filing_date_review.sum()}")
print("\nChart 1 - top agencies by level:")
print(pd.crosstab(df.agency, df.level).assign(total=lambda t: t.sum(axis=1)).sort_values("total", ascending=False).head(10))
print("\nChart 2 - evidence classes:\n", df.evidence_class.value_counts().to_string())
print("\nChart 3 - SOGI phrases (data-element entries):")
print(long.groupby("phrase").agg(entries=("entry_id", "nunique"), mentions=("mention_count", "sum")))
