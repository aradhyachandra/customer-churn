"""Loader tests. These never open the real Excel file: they use tiny fake files and a fake reader."""

import hashlib

import numpy as np
import pandas as pd
import pytest

from churn.load import (
    RawDataChecksumError,
    file_sha256,
    load_raw,
    normalise_sheet,
    verify_checksum,
)

SHEETS = ["Year 2009-2010", "Year 2010-2011"]


def raw_sheet(**overrides):
    """A sheet as the Excel reader returns it (original column names, loose types)."""
    data = {
        "Invoice": [489434, "C489449"],
        "StockCode": [85048, "79323P"],
        "Description": ["LIGHT", "LIGHT"],
        "Quantity": [12, -1],
        "InvoiceDate": pd.to_datetime(["2009-12-01 07:45", "2009-12-01 10:33"]),
        "Price": [6.95, 6.75],
        "Customer ID": [13085.0, np.nan],
        "Country": ["United Kingdom", "United Kingdom"],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def test_file_sha256_matches_hashlib(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"hello world")
    assert file_sha256(f) == hashlib.sha256(b"hello world").hexdigest()


def test_verify_checksum_fails_clearly_on_mismatch(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"hello world")
    with pytest.raises(RawDataChecksumError) as err:
        verify_checksum(f, "0" * 64)
    msg = str(err.value)
    assert "0" * 64 in msg
    assert hashlib.sha256(b"hello world").hexdigest() in msg
    assert str(f) in msg


def test_verify_checksum_passes_on_match(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"hello world")
    verify_checksum(f, hashlib.sha256(b"hello world").hexdigest())


def test_verify_checksum_fails_clearly_when_file_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing.xlsx"):
        verify_checksum(tmp_path / "missing.xlsx", "0" * 64)


def test_customer_id_float_becomes_nullable_int():
    out = normalise_sheet(raw_sheet(), "Year 2009-2010")
    assert str(out["customer_id"].dtype) == "Int64"
    assert out["customer_id"].iloc[0] == 13085
    assert str(out["customer_id"].iloc[0]) == "13085"
    assert pd.isna(out["customer_id"].iloc[1])


def test_customer_id_with_fraction_is_rejected():
    with pytest.raises(ValueError, match="Customer ID"):
        normalise_sheet(raw_sheet(**{"Customer ID": [13085.5, np.nan]}), "s")


def test_codes_become_strings_without_float_suffix():
    df = raw_sheet(Invoice=[489434.0, "C489449"], StockCode=[85048.0, "79323P"])
    out = normalise_sheet(df, "s")
    assert list(out["invoice"]) == ["489434", "C489449"]
    assert list(out["stock_code"]) == ["85048", "79323P"]


def test_columns_renamed_and_sheet_added():
    out = normalise_sheet(raw_sheet(), "Year 2009-2010")
    assert list(out.columns) == [
        "invoice", "stock_code", "description", "quantity",
        "invoice_date", "price", "customer_id", "country", "sheet",
    ]
    assert set(out["sheet"]) == {"Year 2009-2010"}


def test_missing_column_is_rejected():
    with pytest.raises(ValueError, match="Country"):
        normalise_sheet(raw_sheet().drop(columns="Country"), "s")


class FakeReader:
    """Stands in for the Excel reader and counts how often it is called."""

    def __init__(self):
        self.calls = 0

    def __call__(self, path, sheets):
        self.calls += 1
        return {s: raw_sheet() for s in sheets}


def test_load_raw_reads_once_then_uses_cache(tmp_path):
    raw = tmp_path / "raw.xlsx"
    raw.write_bytes(b"pretend excel")
    sha = file_sha256(raw)
    reader = FakeReader()

    first = load_raw(raw, sha, SHEETS, tmp_path / "cache", reader=reader)
    second = load_raw(raw, sha, SHEETS, tmp_path / "cache", reader=reader)

    assert reader.calls == 1
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 4
    assert list(first["sheet"].unique()) == SHEETS
    assert str(second["customer_id"].dtype) == "Int64"


def test_load_raw_checks_checksum_before_using_cache(tmp_path):
    raw = tmp_path / "raw.xlsx"
    raw.write_bytes(b"pretend excel")
    sha = file_sha256(raw)
    load_raw(raw, sha, SHEETS, tmp_path / "cache", reader=FakeReader())

    raw.write_bytes(b"a different file")
    reader = FakeReader()
    with pytest.raises(RawDataChecksumError):
        load_raw(raw, sha, SHEETS, tmp_path / "cache", reader=reader)
    assert reader.calls == 0
