#!/usr/bin/env python3
import pandas as pd
from functools import reduce
import sys

# ─── USER CONFIGURATION ────────────────────────────────────────────────────────
# List all your CSV file paths here:
file_paths = [
    "data/modelTraining/ws1/ws1_3day1.csv",
    "data/modelTraining/ws1/ws1_3day2.csv",
    "data/modelTraining/ws1/ws1_3day3.csv",
    "data/modelTraining/ws1/ws1_3day4.csv",
    # add or remove as needed...
]

# Name of the column in each CSV that contains the timestamp/date:
date_column = "created_at"

# How to merge the data:
#   'outer' -> union of all timestamps (fills missing with NaN)
#   'inner' -> only timestamps common to all files
#   'left'  -> timestamps from the first file, joined with others
join_type = "outer"

# Path for the output merged CSV:
output_file = "data/modelTraining/ws1_3dayT.csv"
# ───────────────────────────────────────────────────────────────────────────────

def load_csv(fp: str) -> pd.DataFrame:
    """Load one day's CSV, parse its date column, and verify schema."""
    try:
        df = pd.read_csv(fp, parse_dates=[date_column])
    except Exception as e:
        print(f"[Error] could not read '{fp}': {e}", file=sys.stderr)
        sys.exit(1)
    if date_column not in df.columns:
        print(f"[Error] '{date_column}' column missing in '{fp}'", file=sys.stderr)
        sys.exit(1)
    return df

def main():
    # 1) Load each file into a list of DataFrames
    dfs = [load_csv(fp) for fp in file_paths]
    if not dfs:
        print("[Error] no files to process—check your 'file_paths'.", file=sys.stderr)
        sys.exit(1)

    # 2) Concatenate them into one DataFrame
    merged = pd.concat(dfs, ignore_index=True)

    # 3) Sort by the timestamp column so rows are chronological
    merged = merged.sort_values(by=date_column)

    # 4) Optional: drop exact duplicate rows (if your data might repeat)
    # merged = merged.drop_duplicates()

    # 5) Write the result out
    try:
        # we write the date column as a normal column (index=False)
        merged.to_csv(output_file, index=False)
        print(f"✅ Successfully wrote {len(merged)} rows to '{output_file}'")
    except Exception as e:
        print(f"[Error] writing '{output_file}': {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()