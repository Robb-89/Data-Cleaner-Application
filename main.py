from __future__ import annotations

import csv
import io
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


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


def map_columns_to_canonical(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    lookup = build_header_lookup(list(df.columns))
    mapping_used: Dict[str, str] = {}
    rename_map = {}

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

    return df.rename(columns=rename_map), mapping_used


def strip_or_nan(x):
    if x is None or pd.isna(x):
        return np.nan
    s = str(x).strip()
    return s if s != "" else np.nan


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
    if not digits:
        return None
    if len(digits) == 11 and digits.startswith(country_code):
        return f"+{digits}"
    if len(digits) == 10:
        return f"+{country_code}{digits}"
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
    dt = pd.to_datetime(x, errors="coerce", infer_datetime_format=True)
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
    items = [e for e in errs if e]
    return ";".join(items) if items else None


# ----------------------------
# Critical: Fix Excel "CSV in one column" sheets
# ----------------------------

def maybe_expand_csv_in_one_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    If Excel sheet has a single column where each cell is a CSV line,
    expand it into proper columns using csv.reader.
    Handles ragged rows by trimming/padding to header length.
    """
    if df.shape[1] != 1:
        return df

    only_col_name = str(df.columns[0])
    if "," not in only_col_name:
        return df

    # Build CSV text: header line + each row as a line
    lines = [only_col_name]
    for v in df.iloc[:, 0].tolist():
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



# ----------------------------
# Loaders
# ----------------------------

def load_from_excel(path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
    return maybe_expand_csv_in_one_column(df)


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


# ----------------------------
# Cleaner
# ----------------------------

def clean_contacts_df(df_raw: pd.DataFrame, opts: CleanerOptions) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df_raw.copy()

    if opts.drop_empty_rows:
        df = df.replace(r"^\s*$", np.nan, regex=True).dropna(how="all")

    df, mapping_used = map_columns_to_canonical(df)
    df["__row_id"] = np.arange(len(df))

    # Ensure canonical columns exist
    for col in ["name_first", "name_last", "email", "phone", "addr_street", "addr_city", "addr_state", "addr_zip",
                "signup_date", "status", "notes"]:
        if col not in df.columns:
            df[col] = np.nan

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

    # Required field errors
    req_err = pd.Series([None] * len(df))
    for field in REQUIRED_FIELDS:
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
        cleaned = apply_dedupe(cleaned, opts.dedupe_mode)

    base_cols = [
        "name_first", "name_last", "full_name",
        "email", "email_domain",
        "phone", "phone_e164",
        "addr_street", "addr_city", "addr_state", "addr_zip",
        "full_address",
        "signup_date", "status", "notes",
    ]

    if opts.keep_original_columns:
        other_cols = [c for c in cleaned.columns if c not in base_cols and c != "__row_id"]
        out_cols = base_cols + other_cols
    else:
        out_cols = base_cols

    cleaned_out = cleaned[out_cols].copy()

    # Mapping report (safe)
    mapping_rows = [{"canonical": k, "source_column": v} for k, v in mapping_used.items()]
    mapping_df = pd.DataFrame(mapping_rows, columns=["canonical", "source_column"]) if mapping_rows else pd.DataFrame(
        columns=["canonical", "source_column"]
    )

    errors_out = df_errors.merge(mapping_df.assign(__row_id=np.nan), how="outer")
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

    # Remove old outputs first (avoids weird partial overwrites)
    safe_remove(cleaned_csv)
    safe_remove(cleaned_xlsx)
    safe_remove(errors_csv)

    cleaned_df.to_csv(cleaned_csv, index=False)
    cleaned_df.to_excel(cleaned_xlsx, index=False)
    errors_df.to_csv(errors_csv, index=False)

    print(f"[OK] Wrote: {cleaned_csv}")
    print(f"[OK] Wrote: {cleaned_xlsx}")
    print(f"[OK] Wrote: {errors_csv}")


# ----------------------------
# CLI
# ----------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python data_cleaner.py excel --path file.xlsx --sheet Sheet1 --out cleaned")
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

    if mode != "excel":
        raise SystemExit("This updated version focuses on fixing your Excel input. Use mode: excel")

    path = get_arg("--path")
    sheet = get_arg("--sheet")
    if not path:
        raise SystemExit("excel mode requires --path <file.xlsx>")

    df_raw = load_from_excel(path, sheet_name=sheet)

    # Quick visibility for debugging
    print("[INFO] Loaded shape:", df_raw.shape)
    print("[INFO] Columns:", list(df_raw.columns))

    cleaned, errors = clean_contacts_df(df_raw, opts)
    export_outputs(cleaned, errors, opts.output_prefix)


if __name__ == "__main__":
    main()
