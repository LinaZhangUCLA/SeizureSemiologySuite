# normalize_times_mmss.py
import argparse
import os
import re
import math
import pandas as pd

def read_one_sheet(path: str, sheet_name=None) -> pd.DataFrame:
    target = sheet_name if sheet_name is not None else 0
    df = pd.read_excel(path, sheet_name=target)
    if isinstance(df, dict):                # rare case: pandas returns all sheets
        first_key = next(iter(df))
        print(f"[WARN] {os.path.basename(path)} has multiple sheets; using first: {first_key}")
        df = df[first_key]
    df.columns = [str(c).strip() for c in df.columns]
    return df

def fmt_mm_ss(total_seconds: int) -> str:
    total_seconds = max(0, int(round(total_seconds)))
    m, s = divmod(total_seconds, 60)
    # zero-padded minutes (always two digits), zero-padded seconds
    return f"{m:02d}:{s:02d}"

def parse_number_like(x):
    """Heuristic for numeric cells:
       - If 0 <= x < 2   -> treat as Excel 'fraction of a day'
       - Else            -> treat as raw seconds
    """
    if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x):
        if 0 <= float(x) < 2:
            seconds = int(round((float(x) % 1.0) * 86400))
            return seconds, seconds >= 3600  # flag if hours present; we’ll drop hours later
        else:
            # assume raw seconds (can be float)
            return int(round(float(x))), float(x) >= 3600
    return None, False

def parse_string_like(s: str):
    """Parse strings to seconds with 'ignore hours' policy and special mm:ss:00 handling."""
    s = s.strip()
    if not s:
        return None, False

    # Special case: looks like mm:ss:ff (we ignore the 3rd group, e.g., '02:25:00' -> 2m25s)
    m = re.fullmatch(r"(\d{1,2}):(\d{2}):(\d{2})", s)
    if m:
        mm, ss, ff = m.groups()
        mm_i, ss_i, ff_i = int(mm), int(ss), int(ff)
        # Treat as minutes:seconds if plausible; discard the 3rd group
        if 0 <= mm_i <= 59 and 0 <= ss_i <= 59:
            return mm_i * 60 + ss_i, False  # no hours ignored in this path
        # Otherwise fall through to generic parsing below

    # Simple mm:ss (or m:ss)
    m2 = re.fullmatch(r"(\d{1,2}):(\d{2})", s)
    if m2:
        mm, ss = m2.groups()
        mm_i, ss_i = int(mm), int(ss)
        if 0 <= mm_i <= 59 and 0 <= ss_i <= 59:
            return mm_i * 60 + ss_i, False
        return None, False

    # Raw seconds as string
    if re.fullmatch(r"\d+(\.\d+)?", s):
        val = float(s)
        return int(round(val)), val >= 3600

    # Try pandas to_timedelta (e.g., '0:34', '00:00:34', '2:03:05')
    try:
        td = pd.to_timedelta(s)
        secs = int(round(td.total_seconds()))
        # Ignore any hours part (you said all < 59 min)
        return secs % 3600, secs >= 3600
    except Exception:
        pass

    # Try time-of-day (e.g., '12:04 AM')
    try:
        ts = pd.to_datetime(s, errors="raise")
        secs = ts.hour * 3600 + ts.minute * 60 + ts.second
        # Ignore hours
        return secs % 3600, ts.hour > 0
    except Exception:
        pass

    return None, False

def to_mm_ss(value):
    """Return (mmss_string or 'N/A', flags_dict)."""
    flags = {"ignored_hours": False, "unparsed": False}

    if pd.isna(value) or (isinstance(value, str) and value.strip() == ""):
        return "N/A", flags

    # pandas-native types
    if isinstance(value, pd.Timestamp):
        secs = value.hour * 3600 + value.minute * 60 + value.second
        flags["ignored_hours"] = value.hour > 0
        return fmt_mm_ss(secs % 3600), flags

    if isinstance(value, pd.Timedelta):
        secs = int(round(value.total_seconds()))
        flags["ignored_hours"] = secs >= 3600
        return fmt_mm_ss(secs % 3600), flags

    # numerics
    secs, had_hours = parse_number_like(value)
    if secs is not None:
        flags["ignored_hours"] = had_hours
        return fmt_mm_ss(secs % 3600), flags

    # strings
    if isinstance(value, str):
        secs, had_hours = parse_string_like(value)
        if secs is not None:
            flags["ignored_hours"] = had_hours
            return fmt_mm_ss(secs % 3600), flags

    # give up
    flags["unparsed"] = True
    return "N/A", flags

def normalize_time_column(series: pd.Series, colname: str):
    parsed = 0
    set_na = 0
    ignored_hours = 0
    unparsed = 0

    out = []
    for _, v in series.items():
        mmss, fl = to_mm_ss(v)
        out.append(mmss)
        if mmss == "N/A":
            set_na += 1
            if fl["unparsed"]:
                unparsed += 1
        else:
            parsed += 1
        if fl["ignored_hours"]:
            ignored_hours += 1

    print(f"[TIME] {colname} -> parsed={parsed}, N/A={set_na}, ignored_hours={ignored_hours}, unparsed={unparsed}")
    return pd.Series(out, index=series.index)

def main():
    ap = argparse.ArgumentParser(description="Normalize all 'time' columns to MM:SS (zero-padded minutes).")
    ap.add_argument("input", help="Input .xlsx path")
    ap.add_argument("-o", "--out", help="Output .xlsx path (default: <input>_mmss.xlsx)")
    ap.add_argument("--sheet", default=None, help="Sheet name (default: first sheet)")
    args = ap.parse_args()

    in_path = args.input
    out_path = args.out or (os.path.splitext(in_path)[0] + "_mmss.xlsx")

    if not os.path.isfile(in_path):
        raise SystemExit(f"[ERROR] File not found: {in_path}")

    df = read_one_sheet(in_path, args.sheet)

    time_cols = [c for c in df.columns if "time" in c.lower()]
    if not time_cols:
        print("[INFO] No 'time' columns found. Writing unchanged copy.")
        df.to_excel(out_path, index=False)
        print(f"[DONE] {out_path}")
        return

    print("[INFO] Time columns:")
    for c in time_cols:
        print(f"  - {c}")

    for c in time_cols:
        df[c] = normalize_time_column(df[c], c)

    df.to_excel(out_path, index=False)
    print(f"[DONE] Saved -> {out_path}")

if __name__ == "__main__":
    main()
