"""Load the raw Online Retail II workbook (both sheets) with a checksum check and a parquet cache.

The Excel file is slow and memory-hungry to read, so it is read once and both sheets are cached
as one parquet file named after the file's checksum. A changed raw file never reuses an old cache.
"""

import hashlib
from pathlib import Path

import pandas as pd

# Original column name -> name used in code.
COLUMNS = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "price",
    "Customer ID": "customer_id",
    "Country": "country",
}


class RawDataChecksumError(Exception):
    """The raw file is not the file the project was built on."""


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_checksum(path, expected):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")
    actual = file_sha256(path)
    if actual != expected:
        raise RawDataChecksumError(
            f"Checksum mismatch for {path}\n"
            f"  expected sha256: {expected}\n"
            f"  actual sha256:   {actual}\n"
            "The raw file differs from the one recorded in config/data.yaml. "
            "Re-download it from UCI (dataset 502) rather than changing the checksum."
        )


def _code_to_str(value):
    """Excel cells can hold 489434, 489434.0 or 'C489449'; return '489434' or 'C489449'."""
    if pd.isna(value):
        return pd.NA
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _to_nullable_int(series, name):
    numbers = pd.to_numeric(series, errors="raise")
    non_null = numbers.dropna()
    if not (non_null == non_null.round()).all():
        raise ValueError(f"{name} has non-integer values, e.g. {non_null[non_null != non_null.round()].iloc[0]}")
    return numbers.astype("Int64")


def normalise_sheet(df, sheet_name):
    """Rename columns, fix types and add the sheet name. Values are not changed otherwise."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Sheet {sheet_name!r} is missing columns: {missing}")
    out = df[list(COLUMNS)].rename(columns=COLUMNS)
    out["invoice"] = out["invoice"].map(_code_to_str).astype("str")
    out["stock_code"] = out["stock_code"].map(_code_to_str).astype("str")
    out["description"] = out["description"].astype("str")
    out["quantity"] = _to_nullable_int(out["quantity"], "Quantity").astype("int64")
    out["invoice_date"] = pd.to_datetime(out["invoice_date"])
    out["price"] = out["price"].astype("float64")
    out["customer_id"] = _to_nullable_int(out["customer_id"], "Customer ID")
    out["country"] = out["country"].astype("str")
    out["sheet"] = sheet_name
    out["sheet"] = out["sheet"].astype("str")
    return out


def read_excel_sheets(path, sheets):
    """Read the named sheets as raw cell values (codes kept as objects so nothing is coerced)."""
    raw_dtypes = {"Invoice": object, "StockCode": object, "Customer ID": object}
    return pd.read_excel(path, sheet_name=list(sheets), engine="calamine", dtype=raw_dtypes)


def load_raw(raw_path, expected_sha256, sheets, cache_dir, reader=read_excel_sheets):
    """Return both sheets stacked in one DataFrame, with a 'sheet' column.

    The checksum is always verified first. The Excel file is read only if no cache exists.
    """
    verify_checksum(raw_path, expected_sha256)
    cache = Path(cache_dir) / f"raw_sheets_{expected_sha256[:12]}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    by_sheet = reader(raw_path, sheets)
    frames = [normalise_sheet(by_sheet[s], s) for s in sheets]
    combined = pd.concat(frames, ignore_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(cache, index=False)
    return pd.read_parquet(cache)
