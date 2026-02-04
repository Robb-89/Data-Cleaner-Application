from __future__ import annotations

import csv
import io
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import signal
from collections import Counter


_STREAMING_SUMMARY_STATE: Optional[dict] = None


def _init_streaming_summary(prefix: str) -> None:
    global _STREAMING_SUMMARY_STATE
    _STREAMING_SUMMARY_STATE = {
        "prefix": prefix,
        "total_in": 0,
        "total_out": 0,
        "invalid_email": 0,
        "invalid_phone": 0,
        "missing_required": 0,
        "error_types": Counter(),
    }


def _update_streaming_summary(
    total_in: int = 0,
    total_out: int = 0,
    invalid_email: int = 0,
    invalid_phone: int = 0,
    missing_required: int = 0,
    error_types: Optional[Counter] = None,
) -> None:
    global _STREAMING_SUMMARY_STATE
    if _STREAMING_SUMMARY_STATE is None:
        return
    _STREAMING_SUMMARY_STATE["total_in"] += int(total_in)
    _STREAMING_SUMMARY_STATE["total_out"] += int(total_out)
    _STREAMING_SUMMARY_STATE["invalid_email"] += int(invalid_email)
    _STREAMING_SUMMARY_STATE["invalid_phone"] += int(invalid_phone)
    _STREAMING_SUMMARY_STATE["missing_required"] += int(missing_required)
    if error_types:
        _STREAMING_SUMMARY_STATE["error_types"].update(error_types)


def _flush_streaming_summary(partial: bool = False) -> None:
    global _STREAMING_SUMMARY_STATE
    if _STREAMING_SUMMARY_STATE is None:
        return
    s = _STREAMING_SUMMARY_STATE
    if partial:
        print("[WARN] Writing partial summary before exit.")
    _write_summary(
        s["prefix"],
        s["total_in"],
        s["total_out"],
        s["invalid_email"],
        s["invalid_phone"],
        s["missing_required"],
        s["error_types"],
    )
    _STREAMING_SUMMARY_STATE = None


def _signal_exit(signum, frame):
    # Attempt to persist partial summary if we're in a streaming run.
    try:
        _flush_streaming_summary(partial=True)
    except Exception:
        pass
    print("\n[WARN] Interrupted by user (signal). Exiting.")
    sys.exit(130)

# Register signal handlers so Ctrl+C / Ctrl+Break exit cleanly with code 130
signal.signal(signal.SIGINT, _signal_exit)
if hasattr(signal, "SIGBREAK"):
    signal.signal(signal.SIGBREAK, _signal_exit)


# ----------------------------
# Configuration
# ----------------------------

REQUIRED_FIELDS = [
    "name_first",
    "name_last",
    "email",
    "phone",
    "addr_street",
    "addr_city",
    "addr_state",
    "addr_zip",
]

FULL_NAME_SYNONYMS = [
    "full_name",
    "fullname",
    "name",
    "customer_name",
    "customer name",
    "contact_name",
    "contact name",
]

COLUMN_SYNONYMS: Dict[str, List[str]] = {
    "name_first": ["first", "first_name", "firstname", "fname", "given", "given_name", "first name"],
    "name_last": ["last", "last_name", "lastname", "lname", "surname", "family_name", "last name"],
    "email": ["email", "email_address", "e-mail", "mail", "emailaddress", "email address"],
    "phone": ["phone", "phone_number", "phonenumber", "mobile", "cell", "tel", "telephone", "phone number"],
    "addr_street": ["street", "street_address", "address", "address1", "addr1", "line1", "street address"],
    "addr_city": ["city", "town"],
    "addr_state": ["state", "province", "region"],
    "addr_zip": ["zip", "zipcode", "postal", "postal_code", "postcode", "zip code"],
    "signup_date": ["signup_date", "signupdate", "signup date", "date", "created", "created_at"],
    "status": ["status"],
    "notes": ["notes"],
}

STATE_FIXES = {
    "california": "CA",
    "new york": "NY",
    "texas": "TX",
    "florida": "FL",
}


@dataclass
class CleanerOptions:
    dedupe_mode: str = "smart"  # "email" | "name_phone" | "smart" | "none"
    assume_country_code: str = "1"
    drop_empty_rows: bool = True
    keep_original_columns: bool = False
    output_prefix: str = "cleaned_output"
    chunksize: Optional[int] = None
    write_xlsx: bool = True
    # Columns to exclude from the cleaned output file(s).
    # Default keeps contact essentials; override via --drop_cols to customize.
    drop_output_cols: List[str] = field(
        default_factory=lambda: ["email_domain", "phone_e164", "full_address", "signup_date", "status", "notes"]
    )


# ----------------------------
# Helpers
# ----------------------------


def normalize_header(h: str) -> str:
    if h is None:
        return ""
    h = str(h).strip().lower()
    h = re.sub(r"[\s\-]+", "_", h)
    h = re.sub(r"[^a-z0-9_]", "", h)
    return h


def build_header_lookup(columns: List[str]) -> Dict[str, str]:
    return {normalize_header(c): c for c in columns}


def build_rename_map(columns: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    lookup = build_header_lookup(list(columns))
    mapping_used: Dict[str, str] = {}
    rename_map: Dict[str, str] = {}

    for canonical, variants in COLUMN_SYNONYMS.items():
        candidates = [canonical] + variants
        found = None
        for v in candidates:
            key = normalize_header(v)
            if key in lookup:
                found = lookup[key]
                break
        if found is not None:
            rename_map[found] = canonical
            mapping_used[canonical] = found

    return rename_map, mapping_used


def map_columns_to_canonical(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    rename_map, mapping_used = build_rename_map(list(df.columns))
    return df.rename(columns=rename_map), mapping_used


def strip_or_nan(x):
    if x is None or pd.isna(x):
        return np.nan
    s = str(x).strip()
    return s if s != "" else np.nan


def find_column_by_synonyms(columns: List[str], synonyms: List[str]) -> Optional[str]:
    lookup = build_header_lookup(columns)
    for s in synonyms:
        key = normalize_header(s)
        if key in lookup:
            return lookup[key]
    return None


def split_full_name(x) -> Tuple[Optional[str], Optional[str]]:
    x = strip_or_nan(x)
    if pd.isna(x):
        return None, None
    s = re.sub(r"\s+", " ", str(x)).strip().strip(" <>\"'")
    if not s:
        return None, None

    if "," in s:
        last, first = [p.strip() for p in s.split(",", 1)]
        return (first or None), (last or None)

    parts = s.split(" ")
    if len(parts) == 1:
        return parts[0], None
    return parts[0], parts[-1]


EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def clean_email(x) -> Tuple[Optional[str], Optional[str]]:
    x = strip_or_nan(x)
    if pd.isna(x):
        return None, "missing_email"
    email = str(x).strip().lower().strip(" <>\"'")
    if not EMAIL_RE.match(email):
        return email, "invalid_email"
    return email, None


def clean_phone(x) -> Tuple[Optional[str], Optional[str]]:
    x = strip_or_nan(x)
    if pd.isna(x):
        return None, "missing_phone"
    digits = re.sub(r"\D+", "", str(x))
    if digits == "":
        return None, "missing_phone"
    if len(digits) not in (7, 10, 11):
        return digits, "invalid_phone_length"
    return digits, None


def phone_to_e164(digits: Optional[str], country_code: str = "1") -> Optional[str]:
    # Pandas may represent missing values as NaN (float), even in "stringy" columns.
    if digits is None or pd.isna(digits):
        return None
    s = str(digits).strip()
    if not s:
        return None
    # `clean_phone()` should already return digits-only, but this keeps us robust.
    s = re.sub(r"\D+", "", s)
    if not s:
        return None
    if len(s) == 11 and s.startswith(country_code):
        return f"+{s}"
    if len(s) == 10:
        return f"+{country_code}{s}"
    return None


def clean_state(x) -> Tuple[Optional[str], Optional[str]]:
    x = strip_or_nan(x)
    if pd.isna(x):
        return None, "missing_state"
    s = re.sub(r"\s+", " ", str(x).strip().lower())
    if s in STATE_FIXES:
        return STATE_FIXES[s], None
    if len(s) == 2 and s.isalpha():
        return s.upper(), None
    if s.isalpha() and len(s) > 2:
        return s.title(), "non_standard_state"
    return str(x).strip(), "invalid_state"


def clean_zip(x) -> Tuple[Optional[str], Optional[str]]:
    x = strip_or_nan(x)
    if pd.isna(x):
        return None, "missing_zip"
    digits = re.sub(r"\D+", "", str(x))
    if digits == "":
        return None, "missing_zip"
    if len(digits) < 5:
        return digits.zfill(5), "zip_padded"
    return digits[:5], None


def parse_date_to_iso(x) -> Tuple[Optional[str], Optional[str]]:
    x = strip_or_nan(x)
    if pd.isna(x):
        return None, None
    # Some pandas versions don't accept `infer_datetime_format`; keep it simple and robust.
    dt = pd.to_datetime(x, errors="coerce")
    if pd.isna(dt):
        return str(x), "invalid_date"
    return dt.date().isoformat(), None


def title_case_name(x) -> Optional[str]:
    x = strip_or_nan(x)
    if pd.isna(x):
        return None
    s = re.sub(r"\s+", " ", str(x)).strip()
    parts = []
    for p in s.split(" "):
        if "'" in p:
            parts.append("'".join(w.capitalize() for w in p.split("'")))
        else:
            parts.append(p.capitalize())
    return " ".join(parts)


def build_full_name(first: Optional[str], last: Optional[str]) -> Optional[str]:
    if not first and not last:
        return None
    return f"{first or ''} {last or ''}".strip() or None


def build_full_address(full_name, street, city, state, zip_code) -> Optional[str]:
    def s(x) -> str:
        if x is None or pd.isna(x):
            return ""
        return str(x).strip()

    line1 = s(full_name)
    line2 = s(street)
    city_s = s(city)
    state_s = s(state)
    zip_s = s(zip_code)

    if city_s and state_s:
        line3 = f"{city_s}, {state_s} {zip_s}".strip()
    else:
        line3 = " ".join(p for p in [city_s, state_s, zip_s] if p).strip()

    if not any([line1, line2, line3]):
        return None

    return "\n".join([line1, line2, line3]).rstrip("\n")


def combine_errors(*errs: Optional[str]) -> Optional[str]:
    # Errors may be None/NaN depending on pandas operations; normalize to strings.
    items: List[str] = []
    for e in errs:
        if e is None or pd.isna(e):
            continue
        s = str(e).strip()
        if not s:
            continue
        items.append(s)
    return ";".join(items) if items else None


# ----------------------------
# Critical: Fix Excel "CSV in one column" sheets
# ----------------------------


def maybe_expand_csv_in_one_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    If Excel sheet has a single column where each cell is a CSV line,
    expand it into proper columns using csv.reader.
    Handles ragged rows by trimming/padding to header length.

    This is more robust than checking only the column name: some sheets
    put the entire CSV (header + rows) into the *first cell* or the column header.
    """
    if df.shape[1] != 1:
        return df

    # Get the header (column name) and the first few cell values.
    only_col_name = str(df.columns[0])
    values = df.iloc[:, 0].astype(str).tolist()

    header_line = None
    if "," in only_col_name:
        header_line = only_col_name
        data_values = values  # when header is in the column name, include all rows
    else:
        # look for a row that looks like a CSV header (has commas and likely non-numeric fields)
        start_row = None
        for i, v in enumerate(values):
            if v is None:
                continue
            s = str(v).strip()
            if "," in s:
                start_row = i
                header_line = s
                break
        if start_row is None:
            return df
        data_values = values[start_row + 1 :]

    # Build CSV text: header line + each data row as a line
    lines = [header_line]
    for v in data_values:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        lines.append(str(v))

    csv_text = "\n".join(lines)

    reader = csv.reader(io.StringIO(csv_text), skipinitialspace=False)
    rows = list(reader)
    if not rows:
        return df

    header = rows[0]
    data = rows[1:]
    n = len(header)

    fixed_rows = []
    for r in data:
        if len(r) == n:
            fixed_rows.append(r)
        elif len(r) > n:
            # Too many fields: keep extras joined into the last column
            fixed_rows.append(r[: n - 1] + [",".join(r[n - 1 :])])
        else:
            # Too few fields: pad with blanks
            fixed_rows.append(r + [""] * (n - len(r)))

    expanded = pd.DataFrame(fixed_rows, columns=header)
    return expanded


def looks_like_csv_in_one_column(df: pd.DataFrame) -> bool:
    if df.shape[1] != 1:
        return False
    only_col_name = str(df.columns[0])
    if "," in only_col_name:
        return True
    values = df.iloc[:, 0].astype(str).tolist()
    for v in values[:50]:
        if v is None:
            continue
        s = str(v).strip()
        if "," in s:
            return True
    return False



# ----------------------------
# Loaders
# ----------------------------


def load_from_excel(path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    try:
        df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
    except ModuleNotFoundError:
        raise SystemExit("Missing required package 'openpyxl'. Install it with: pip install openpyxl")
    return maybe_expand_csv_in_one_column(df)


def load_from_csv(path: str, chunksize: Optional[int] = None):
    def robust_csv_reader(path, chunksize=None):
        with open(path, newline='', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            rows = list(reader)
        if not rows:
            return pd.DataFrame()
        header = rows[0]
        n = len(header)
        def fix_row(r):
            if len(r) == n:
                return r
            elif len(r) > n:
                return r[:n-1] + [','.join(r[n-1:])]
            else:
                return r + [''] * (n - len(r))
        fixed_rows = [fix_row(r) for r in rows[1:]]
        df = pd.DataFrame(fixed_rows, columns=header)
        return df

    if chunksize:
        # For chunked mode, read all, fix, then yield in chunks
        df = robust_csv_reader(path)
        for i in range(0, len(df), chunksize):
            yield df.iloc[i:i+chunksize].reset_index(drop=True)
    else:
        df = robust_csv_reader(path)
        return maybe_expand_csv_in_one_column(df)


def count_csv_rows(path: str) -> int:
    """Fast line-count for CSV files. Returns number of data rows (excluding header) or 0 on error."""
    try:
        # Open in text mode with errors ignored (robust to minor encoding issues)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            # Count lines; subtract 1 for header if present
            n = sum(1 for _ in f)
        return max(0, n - 1)
    except Exception:
        return 0


# ----------------------------
# Dedupe
# ----------------------------


def apply_dedupe(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    d = df.copy()
    if mode == "email":
        d["_k"] = d["email"].fillna("")
    elif mode == "name_phone":
        d["_k"] = (d["full_name"].fillna("") + "|" + d["phone"].fillna(""))
    elif mode == "smart":
        email_key = d["email"].fillna("")
        fallback = (d["full_name"].fillna("") + "|" + d["phone"].fillna(""))
        d["_k"] = np.where(email_key != "", "email:" + email_key, "np:" + fallback)
    else:
        return d

    has_key = d["_k"].astype(str).str.len() > 0
    out = pd.concat([d[~has_key], d[has_key].drop_duplicates("_k")], axis=0).drop(columns=["_k"])
    return out.reset_index(drop=True)


def apply_dedupe_stream(df: pd.DataFrame, mode: str, seen_keys: set) -> pd.DataFrame:
    if mode == "email":
        keys = df["email"].fillna("").astype(str).tolist()
    elif mode == "name_phone":
        keys = (df["full_name"].fillna("") + "|" + df["phone"].fillna("")).astype(str).tolist()
    elif mode == "smart":
        email_key = df["email"].fillna("").astype(str)
        fallback = (df["full_name"].fillna("") + "|" + df["phone"].fillna("")).astype(str)
        keys = np.where(email_key != "", "email:" + email_key, "np:" + fallback).tolist()
    else:
        return df

    keep_mask = []
    for k in keys:
        if not k:
            keep_mask.append(True)
            continue
        if k in seen_keys:
            keep_mask.append(False)
        else:
            keep_mask.append(True)
            seen_keys.add(k)

    return df.loc[keep_mask].reset_index(drop=True)


# ----------------------------
# Cleaner
# ----------------------------


def clean_contacts_df(
    df_raw: pd.DataFrame,
    opts: CleanerOptions,
    rename_map: Optional[Dict[str, str]] = None,
    mapping_used: Optional[Dict[str, str]] = None,
    include_mapping: bool = True,
    dedupe_state: Optional[set] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df_raw.copy()

    if opts.drop_empty_rows:
        df = df.replace(r"^\s*$", np.nan, regex=True).dropna(how="all")

    if rename_map is None or mapping_used is None:
        df, mapping_used = map_columns_to_canonical(df)
    else:
        df = df.rename(columns=rename_map)
    df["__row_id"] = np.arange(len(df))

    # Ensure canonical columns exist
    # Always ensure these canonical columns exist, even if not present in input
    for col in ["name_first", "name_last", "email", "phone", "addr_street", "addr_city", "addr_state", "addr_zip",
                "signup_date", "status", "notes"]:
        if col not in df.columns:
            df[col] = np.nan

    # If input provides a single full-name column (e.g. "customer_name"), split it into first/last.
    # Do this before title-casing so we get consistent output.
    full_name_col = find_column_by_synonyms(list(df.columns), FULL_NAME_SYNONYMS)
    if full_name_col:
        parsed = df[full_name_col].apply(split_full_name)
        first_from_full = parsed.apply(lambda t: t[0])
        last_from_full = parsed.apply(lambda t: t[1])
        df["name_first"] = df["name_first"].combine_first(first_from_full)
        df["name_last"] = df["name_last"].combine_first(last_from_full)

    df["name_first"] = df["name_first"].apply(title_case_name)
    df["name_last"] = df["name_last"].apply(title_case_name)

    df["addr_street"] = df["addr_street"].apply(strip_or_nan)
    df["addr_city"] = df["addr_city"].apply(lambda x: title_case_name(x))

    st = df["addr_state"].apply(clean_state)
    df["addr_state"] = st.apply(lambda t: t[0])
    st_err = st.apply(lambda t: t[1])

    z = df["addr_zip"].apply(clean_zip)
    df["addr_zip"] = z.apply(lambda t: t[0])
    zip_err = z.apply(lambda t: t[1])

    e = df["email"].apply(clean_email)
    df["email"] = e.apply(lambda t: t[0])
    email_err = e.apply(lambda t: t[1])

    p = df["phone"].apply(clean_phone)
    df["phone"] = p.apply(lambda t: t[0])
    phone_err = p.apply(lambda t: t[1])

    d = df["signup_date"].apply(parse_date_to_iso)
    df["signup_date"] = d.apply(lambda t: t[0])
    date_err = d.apply(lambda t: t[1])

    df["status"] = df["status"].apply(strip_or_nan).apply(lambda x: str(x).strip().lower() if not pd.isna(x) else np.nan)
    df["notes"] = df["notes"].apply(strip_or_nan)

    df["full_name"] = df.apply(lambda r: build_full_name(r["name_first"], r["name_last"]), axis=1)
    df["email_domain"] = df["email"].apply(lambda v: v.split("@", 1)[1] if isinstance(v, str) and "@" in v else np.nan)
    df["phone_e164"] = df["phone"].apply(lambda v: phone_to_e164(v, opts.assume_country_code))
    df["full_address"] = df.apply(lambda r: build_full_address(
        r["full_name"], r["addr_street"], r["addr_city"], r["addr_state"], r["addr_zip"]
    ), axis=1)

    # Determine applicable required fields (only those present in source or derivable)
    mapping_used = mapping_used or {}
    has_full_name = bool(full_name_col)

    applicable_required = set(k for k in mapping_used.keys() if k in REQUIRED_FIELDS)
    if has_full_name:
        applicable_required.update({"name_first", "name_last"})

    # Required field errors (only for applicable fields)
    req_err = pd.Series([None] * len(df))
    for field in sorted(applicable_required):
        req_err = req_err.combine(df[field].isna().map(lambda m: f"missing_{field}" if m else None),
                                  lambda a, b: combine_errors(a, b))

    df_errors = pd.DataFrame({
        "__row_id": df["__row_id"],
        "error_email": email_err,
        "error_phone": phone_err,
        "error_state": st_err,
        "error_zip": zip_err,
        "error_date": date_err,
        "error_required": req_err,
    })
    df_errors["errors_all"] = df_errors.apply(lambda r: combine_errors(
        r["error_email"], r["error_phone"], r["error_state"], r["error_zip"], r["error_date"], r["error_required"]
    ), axis=1)

    cleaned = df.copy()
    if opts.dedupe_mode != "none":
        if dedupe_state is None:
            cleaned = apply_dedupe(cleaned, opts.dedupe_mode)
        else:
            cleaned = apply_dedupe_stream(cleaned, opts.dedupe_mode, dedupe_state)

    # Determine which base columns to include in output based on detected/derivable fields
    base_cols = []
    # Names
    has_name = has_full_name or ("name_first" in mapping_used) or ("name_last" in mapping_used)
    if has_name:
        base_cols.extend(["name_first", "name_last", "full_name"])
    # Email
    if "email" in mapping_used:
        base_cols.extend(["email", "email_domain"])
    # Phone
    if "phone" in mapping_used:
        base_cols.extend(["phone", "phone_e164"])
    # Address parts
    addr_parts = [p for p in ["addr_street", "addr_city", "addr_state", "addr_zip"] if p in mapping_used]
    base_cols.extend(addr_parts)
    if addr_parts:
        base_cols.append("full_address")
    # Misc
    if "signup_date" in mapping_used:
        base_cols.append("signup_date")
    if "status" in mapping_used:
        base_cols.append("status")
    if "notes" in mapping_used:
        base_cols.append("notes")

    if opts.keep_original_columns:
        other_cols = [c for c in cleaned.columns if c not in base_cols and c != "__row_id"]
        out_cols = base_cols + other_cols
    else:
        out_cols = base_cols

    drop_cols = set((opts.drop_output_cols or []))
    if drop_cols:
        out_cols = [c for c in out_cols if c not in drop_cols]

    cleaned_out = cleaned[out_cols].copy()

    # Mapping report (safe)
    mapping_rows = [{"canonical": k, "source_column": v} for k, v in mapping_used.items()]
    mapping_df = pd.DataFrame(mapping_rows, columns=["canonical", "source_column"]) if mapping_rows else pd.DataFrame(
        columns=["canonical", "source_column"]
    )

    if include_mapping:
        errors_out = df_errors.merge(mapping_df.assign(__row_id=np.nan), how="outer")
    else:
        if "canonical" not in df_errors.columns:
            df_errors["canonical"] = np.nan
        if "source_column" not in df_errors.columns:
            df_errors["source_column"] = np.nan
        errors_out = df_errors
    return cleaned_out, errors_out


# ----------------------------
# Export (handles locked files)
# ----------------------------


def safe_remove(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except PermissionError:
        raise PermissionError(f"File is locked (likely open in Excel): {path}. Close it and re-run.")


def export_outputs(cleaned_df: pd.DataFrame, errors_df: pd.DataFrame, prefix: str) -> None:
    cleaned_csv = f"{prefix}.csv"
    cleaned_xlsx = f"{prefix}.xlsx"
    errors_csv = f"{prefix}_errors.csv"
    missing_report_csv = f"{prefix}_missing_report.csv"

    # Remove old outputs first (avoids weird partial overwrites)
    safe_remove(cleaned_csv)
    safe_remove(cleaned_xlsx)
    safe_remove(errors_csv)
    safe_remove(missing_report_csv)

    cleaned_df.to_csv(cleaned_csv, index=False)
    try:
        cleaned_df.to_excel(cleaned_xlsx, index=False)
        print(f"[OK] Wrote: {cleaned_xlsx}")
    except ModuleNotFoundError:
        print("[WARN] Skipped writing .xlsx because 'openpyxl' is not installed. Install it with: pip install openpyxl")
    errors_df.to_csv(errors_csv, index=False)

    # Write detailed missing/invalid report
    if not errors_df.empty:
        # Choose identifying columns (name, email, phone, row_id if present)
        id_cols = [c for c in ["name_first", "name_last", "full_name", "email", "phone", "__row_id"] if c in errors_df.columns]
        # Find which error columns exist
        error_cols = [c for c in ["error_email", "error_phone", "error_required"] if c in errors_df.columns]
        # For each row, list which fields are missing/invalid
        def error_fields(row):
            fields = []
            for c in error_cols:
                val = str(row.get(c, ""))
                if val and val != "nan" and val != "None":
                    fields.append(f"{c.replace('error_','')}: {val}")
            return "; ".join(fields)
        report_df = errors_df[id_cols].copy()
        report_df["missing_or_invalid_fields"] = errors_df.apply(error_fields, axis=1)
        report_df = report_df[ id_cols + ["missing_or_invalid_fields"] ]
        report_df.to_csv(missing_report_csv, index=False)
        print(f"[OK] Wrote: {missing_report_csv}")

    print(f"[OK] Wrote: {cleaned_csv}")
    print(f"[OK] Wrote: {errors_csv}")


def _compute_error_stats(errors_df: pd.DataFrame):
    """Return (invalid_email_count, invalid_phone_count, missing_required_count, Counter of error types)."""
    invalid_email = int(errors_df["error_email"].fillna("").astype(str).str.contains("invalid_email").sum())
    invalid_phone = int(errors_df["error_phone"].fillna("").astype(str).str.contains("invalid").sum())
    missing_required = int(errors_df["error_required"].fillna("").astype(str).str.contains("missing_").sum())

    c = Counter()
    for s in errors_df["errors_all"].fillna(""):
        for part in [p for p in str(s).split(";") if p]:
            c[part] += 1
    return invalid_email, invalid_phone, missing_required, c


def _write_summary(prefix: str, total_in: int, total_out: int, invalid_email: int, invalid_phone: int, missing_required: int, top_errors: Counter) -> None:
    summary_txt = f"{prefix}_summary.txt"
    lines = []
    lines.append(f"Total rows in: {total_in}")
    lines.append(f"Rows out after dedupe: {total_out}")
    lines.append(f"Invalid email count: {invalid_email}")
    lines.append(f"Invalid phone count: {invalid_phone}")
    lines.append(f"Missing required count: {missing_required}")
    lines.append("")
    lines.append("Top error types:")
    for err, cnt in top_errors.most_common(10):
        lines.append(f"  {err}: {cnt}")
    txt = "\n".join(lines)
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"[OK] Wrote summary: {summary_txt}")


def export_outputs_streaming(
    cleaned_df: pd.DataFrame,
    errors_df: pd.DataFrame,
    prefix: str,
    is_first_chunk: bool,
) -> None:
    cleaned_csv = f"{prefix}.csv"
    errors_csv = f"{prefix}_errors.csv"

    cleaned_df.to_csv(cleaned_csv, index=False, mode="w" if is_first_chunk else "a", header=is_first_chunk)
    errors_df.to_csv(errors_csv, index=False, mode="w" if is_first_chunk else "a", header=is_first_chunk)


# ----------------------------
# CLI
# ----------------------------


def main():
    if len(sys.argv) < 2:
        print("Usage: python data_cleaner.py excel|csv --path file.xlsx --sheet Sheet1 --out cleaned")
        sys.exit(1)

    mode = sys.argv[1].lower()
    opts = CleanerOptions()
    args = sys.argv[2:]

    def get_arg(flag: str, default: Optional[str] = None) -> Optional[str]:
        if flag in args:
            i = args.index(flag)
            if i + 1 < len(args):
                return args[i + 1]
        return default

    opts.output_prefix = get_arg("--out", opts.output_prefix) or opts.output_prefix
    opts.dedupe_mode = get_arg("--dedupe", opts.dedupe_mode) or opts.dedupe_mode
    opts.keep_original_columns = (get_arg("--keep_original", "false") or "false").lower() == "true"
    chunksize_arg = get_arg("--chunksize")
    opts.chunksize = int(chunksize_arg) if chunksize_arg else None
    opts.write_xlsx = (get_arg("--no_xlsx", "false") or "false").lower() != "true"
    drop_cols_arg = get_arg("--drop_cols")
    if drop_cols_arg:
        opts.drop_output_cols = [c.strip() for c in re.split(r"[;,]", drop_cols_arg) if c.strip()]

    if mode not in ("excel", "csv"):
        raise SystemExit("Mode must be 'excel' or 'csv'.")

    path = get_arg("--path")
    sheet = get_arg("--sheet")
    if not path:
        raise SystemExit(f"{mode} mode requires --path <file>")

    if mode == "excel" and opts.chunksize:
        print("[WARN] --chunksize is only supported for CSV. Falling back to full Excel load.")
        opts.chunksize = None

    # Read file
    try:
        if mode == "excel":
            df_raw = load_from_excel(path, sheet_name=sheet)
            # Quick visibility for debugging
            print("[INFO] Loaded shape:", df_raw.shape)
            print("[INFO] Columns:", list(df_raw.columns))
            cleaned, errors = clean_contacts_df(df_raw, opts)
            export_outputs(cleaned, errors, opts.output_prefix)
            inv_e, inv_p, miss_req, top = _compute_error_stats(errors)
            _write_summary(opts.output_prefix, len(df_raw), len(cleaned), inv_e, inv_p, miss_req, top)
            return

        if opts.chunksize:
            chunks = load_from_csv(path, chunksize=opts.chunksize)
        else:
            df_raw = load_from_csv(path)
    except Exception as exc:
        raise SystemExit(f"Failed reading input: {exc}")

    if not opts.chunksize:
        # Quick visibility for debugging
        print("[INFO] Loaded shape:", df_raw.shape)
        print("[INFO] Columns:", list(df_raw.columns))
        cleaned, errors = clean_contacts_df(df_raw, opts)
        export_outputs(cleaned, errors, opts.output_prefix)
        inv_e, inv_p, miss_req, top = _compute_error_stats(errors)
        _write_summary(opts.output_prefix, len(df_raw), len(cleaned), inv_e, inv_p, miss_req, top)
        return

    if opts.write_xlsx:
        print("[WARN] Skipping .xlsx output in chunked mode. Use full mode if you need .xlsx.")

    cleaned_csv = f"{opts.output_prefix}.csv"
    errors_csv = f"{opts.output_prefix}_errors.csv"
    safe_remove(cleaned_csv)
    safe_remove(errors_csv)

    seen_keys: Optional[set] = set() if opts.dedupe_mode != "none" else None
    rename_map = None
    mapping_used = None
    is_first_chunk = True

    # Streaming summary accumulators
    total_in = 0
    total_out = 0
    invalid_email_total = 0
    invalid_phone_total = 0
    missing_required_total = 0
    error_types_total: Counter = Counter()

    # Count total rows (fast single pass) so we can show percentage progress. If we can't determine
    # the count, we'll still display rows processed instead of a percent.
    total_rows = 0
    if mode == "csv":
        total_rows = count_csv_rows(path)
        if total_rows:
            print(f"[INFO] Total rows: {total_rows}")

    processed_rows = 0

    # Initialize global streaming summary state so interrupts can flush partial results
    _init_streaming_summary(opts.output_prefix)

    for chunk in chunks:
        if is_first_chunk and looks_like_csv_in_one_column(chunk):
            print("[WARN] Detected 'CSV in one column' format. Falling back to full-file read.")
            df_raw = load_from_csv(path)
            print("[INFO] Loaded shape:", df_raw.shape)
            print("[INFO] Columns:", list(df_raw.columns))
            cleaned, errors = clean_contacts_df(df_raw, opts)
            export_outputs(cleaned, errors, opts.output_prefix)
            inv_e, inv_p, miss_req, top = _compute_error_stats(errors)
            _write_summary(opts.output_prefix, len(df_raw), len(cleaned), inv_e, inv_p, miss_req, top)
            return

        chunk = maybe_expand_csv_in_one_column(chunk)
        if is_first_chunk:
            rename_map, mapping_used = build_rename_map(list(chunk.columns))

        cleaned_chunk, errors_chunk = clean_contacts_df(
            chunk,
            opts,
            rename_map=rename_map,
            mapping_used=mapping_used,
            include_mapping=False,
            dedupe_state=seen_keys,
        )
        export_outputs_streaming(cleaned_chunk, errors_chunk, opts.output_prefix, is_first_chunk)

        # Update streaming summary accumulators
        total_in += len(chunk)
        total_out += len(cleaned_chunk)
        inv_e, inv_p, miss_req, top = _compute_error_stats(errors_chunk)
        invalid_email_total += inv_e
        invalid_phone_total += inv_p
        missing_required_total += miss_req
        error_types_total.update(top)

        # Mirror updates to global streaming summary state (so we can flush on interrupt)
        _update_streaming_summary(
            total_in=len(chunk),
            total_out=len(cleaned_chunk),
            invalid_email=inv_e,
            invalid_phone=inv_p,
            missing_required=miss_req,
            error_types=top,
        )

        # Update processed count and print progress
        processed_rows += len(chunk)
        if total_rows:
            pct = (processed_rows / total_rows) * 100 if total_rows else 0.0
            print(f"[PROGRESS] {processed_rows}/{total_rows} rows ({pct:.2f}%) — out={total_out} rows — invalid_email={invalid_email_total} invalid_phone={invalid_phone_total} missing_required={missing_required_total}")
        else:
            print(f"[PROGRESS] {processed_rows} rows processed — out={total_out} rows — invalid_email={invalid_email_total} invalid_phone={invalid_phone_total} missing_required={missing_required_total}")

        if is_first_chunk:
            print("[INFO] Columns:", list(chunk.columns))
        is_first_chunk = False

    if mapping_used:
        mapping_rows = [{"canonical": k, "source_column": v} for k, v in mapping_used.items()]
        mapping_df = pd.DataFrame(mapping_rows, columns=["canonical", "source_column"])
        mapping_df["__row_id"] = np.nan
        mapping_df["error_email"] = np.nan
        mapping_df["error_phone"] = np.nan
        mapping_df["error_state"] = np.nan
        mapping_df["error_zip"] = np.nan
        mapping_df["error_date"] = np.nan
        mapping_df["error_required"] = np.nan
        mapping_df["errors_all"] = np.nan
        mapping_df = mapping_df[
            [
                "__row_id",
                "error_email",
                "error_phone",
                "error_state",
                "error_zip",
                "error_date",
                "error_required",
                "errors_all",
                "canonical",
                "source_column",
            ]
        ]
        mapping_df.to_csv(errors_csv, index=False, mode="a", header=False)

    print(f"[OK] Wrote: {cleaned_csv}")
    print(f"[OK] Wrote: {errors_csv}")

    # Final summary for streaming (chunked) mode
    _flush_streaming_summary(partial=False)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        try:
            _flush_streaming_summary(partial=True)
        except Exception:
            pass
        print("\n[WARN] Interrupted by user (Ctrl+C). Exiting.")
        sys.exit(130)
