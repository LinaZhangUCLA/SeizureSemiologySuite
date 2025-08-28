# align_collect_columns_dedupe.py
import os, sys, glob
import pandas as pd
from collections import Counter

DIR = "/root/projects/ucla/data"     # <-- change me
OLD = "/root/projects/ucla/seizure_feature.xlsx"        # <-- change me
OUT = "combined_aligned.xlsx"         # <-- change me
SHEET = None  # e.g., "Sheet1"; if None, use first sheet (index 0)

# How to handle duplicate keys in an input file:
#   'first' -> keep first occurrence
#   'last'  -> keep last occurrence
#   'error' -> raise and stop
DEDUPLICATE_KEY_POLICY = 'first'

def read_one_sheet(path: str, sheet_name=None) -> pd.DataFrame:
    target = sheet_name if sheet_name is not None else 0
    df = pd.read_excel(path, sheet_name=target)
    if isinstance(df, dict):
        first_key = next(iter(df))
        print(f"[WARN] {os.path.basename(path)} has multiple sheets; using first: {first_key}")
        df = df[first_key]
    df.columns = [str(c).strip() for c in df.columns]
    return df

def report_duplicate_columns(df: pd.DataFrame, label: str):
    counts = Counter(df.columns)
    dups = {c:n for c,n in counts.items() if n > 1}
    if dups:
        print(f"[DUP COLS] {label}: duplicate column names found -> {dups}")

def report_duplicate_keys(df: pd.DataFrame, key_col: str, label: str):
    if key_col not in df.columns:
        print(f"[WARN] {label}: missing key column '{key_col}', cannot check duplicates.")
        return False
    keys = df[key_col].astype(str)
    dup_mask = keys.duplicated(keep=False)
    if dup_mask.any():
        # list duplicates with counts
        dup_vals = keys[dup_mask]
        cnts = Counter(dup_vals)
        print(f"[DUP KEYS] {label}: duplicate key values in '{key_col}':")
        # show a compact preview; print all if small
        for k, n in cnts.items():
            print(f"  - {k!r}: {n} occurrences")
        return True
    return False

def dedupe_by_policy(df: pd.DataFrame, key_col: str, label: str):
    if key_col not in df.columns:
        return df
    keys = df[key_col].astype(str)
    if not keys.duplicated(keep=False).any():
        return df  # nothing to do
    if DEDUPLICATE_KEY_POLICY == 'first':
        print(f"[ACTION] {label}: dropping duplicate keys (keep='first').")
        return df.drop_duplicates(subset=[key_col], keep='first')
    elif DEDUPLICATE_KEY_POLICY == 'last':
        print(f"[ACTION] {label}: dropping duplicate keys (keep='last').")
        return df.drop_duplicates(subset=[key_col], keep='last')
    else:
        raise SystemExit(f"[ERROR] {label}: duplicate keys detected and policy is 'error'.")

def main():
    # 1) Load old file (template)
    old_df = read_one_sheet(OLD, SHEET)
    desired_cols = list(old_df.columns)
    if not desired_cols:
        raise SystemExit("[ERROR] Old file has no columns.")
    if len(desired_cols) != len(set(desired_cols)):
        report_duplicate_columns(old_df, "OLD FILE")
        raise SystemExit("[ERROR] Old file has duplicate column names; fix the header first.")

    key_col = desired_cols[0]
    if key_col not in old_df.columns:
        raise SystemExit(f"[ERROR] Key column '{key_col}' not found in old file headers.")

    # Keep old row order and key list
    old_df[key_col] = old_df[key_col].astype(str)
    target_index = old_df[key_col].tolist()

    # 2) Input files (no sort), exclude OLD if in same directory
    files = [f for f in glob.glob(os.path.join(DIR, "*.xlsx"))
             if os.path.abspath(f) != os.path.abspath(OLD)]
    if not files:
        raise SystemExit(f"[ERROR] No .xlsx files found in: {DIR}")
    if len(files) != 8:
        print(f"[WARN] Found {len(files)} .xlsx files (expected 8). Proceeding...")

    # 3) Prepare output frame with same shape/order as old
    out_df = pd.DataFrame(index=target_index, columns=desired_cols)
    out_df[key_col] = target_index

    provided_any = set()

    # 4) Process each input file with duplicate checks
    for f in files:
        label = os.path.basename(f)
        df_i = read_one_sheet(f, SHEET)

        # report duplicate columns
        report_duplicate_columns(df_i, label)

        # must have key column
        if key_col not in df_i.columns:
            print(f"[WARN] {label}: missing key column '{key_col}'. Skipping this file.")
            continue

        # normalize key to str
        df_i[key_col] = df_i[key_col].astype(str)

        # report duplicate keys (values in the first column)
        had_dup_keys = report_duplicate_keys(df_i, key_col, label)
        if had_dup_keys:
            df_i = dedupe_by_policy(df_i, key_col, label)

        # align rows by target_index
        df_i = df_i.set_index(key_col)
        # IMPORTANT: reindex can still fail if index has become duplicated somehow.
        # The dedupe step should prevent that; if it still fails, raise with context.
        try:
            df_i = df_i.reindex(target_index)
        except ValueError as e:
            raise SystemExit(f"[ERROR] {label}: reindex failed even after deduping keys. {e}")

        # fill columns that overlap with desired header (excluding the key)
        overlap = [c for c in desired_cols[1:] if c in df_i.columns]
        provided_any.update(overlap)
        for c in overlap:
            src = df_i[c]
            mask = out_df[c].isna() & src.notna()
            out_df.loc[mask, c] = src[mask]

    # 5) Sizes and warnings
    old_shape = (len(old_df.index), len(desired_cols))
    new_shape = (len(out_df.index), len(out_df.columns))
    print(f"[SIZE] Old file shape: {old_shape}")
    print(f"[SIZE] New file shape: {new_shape}")

    missing_globally = [c for c in desired_cols[1:] if c not in provided_any]
    if missing_globally:
        print("[WARNING] Columns in OLD header not found in ANY input file:")
        print(missing_globally)

    # any rows entirely NaN aside from key?
    all_nan_rows = out_df[desired_cols[1:]].isna().all(axis=1)
    if all_nan_rows.any():
        missing_keys = out_df.loc[all_nan_rows, key_col].tolist()
        print(f"[WARNING] These keys had no data in any input (only the key present): {missing_keys}")

    # 6) Save
    out_df.to_excel(OUT, index=False)
    print(f"[DONE] Saved merged file to: {OUT}")

if __name__ == "__main__":
    main()
