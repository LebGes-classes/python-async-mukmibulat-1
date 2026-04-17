import pytest
import pandas as pd
from datetime import datetime
from hwasync import (
    parse_date,
    normalize_status,
    clean_uptime,
    check_dates,
    filter_warranty
)


def test_parse_date_valid():
    assert parse_date("2026-04-17") == datetime(2026, 4, 17)
    assert parse_date("17.04.2026") == datetime(2026, 4, 17)
    assert parse_date(None) is None


def test_parse_date_invalid():
    assert parse_date("не дата") is None
    assert parse_date("31.31.2026") is None


def test_normalize_status():
    assert normalize_status("OK") == "operational"
    assert normalize_status("broken") == "faulty"
    assert normalize_status("Unknown Status") == "unknown status"
    assert normalize_status(float('nan')) == "unknown"


def test_clean_uptime():
    df = pd.DataFrame({"uptime_pct": ["98,5", "95.0", "invalid"]})
    result_df = clean_uptime(df)

    assert result_df.loc[0, "uptime_pct"] == 98.5
    assert result_df.loc[1, "uptime_pct"] == 95.0
    assert pd.isna(result_df.loc[2, "uptime_pct"])


def test_check_dates_logic():
    df = pd.DataFrame({
        "install_date": [datetime(2026, 1, 1)],
        "last_calibration_date": [datetime(2025, 12, 31)]
    })
    result_df = check_dates(df)

    assert pd.isna(result_df.loc[0, "last_calibration_date"])


def test_filter_warranty():
    df = pd.DataFrame({
        "device_id": [1, 2],
        "warranty_until": [
            datetime(2027, 12, 31),
            datetime(2025, 12, 31)
        ]
    })
    in_w, out_w = filter_warranty(df)

    assert len(in_w) == 1
    assert len(out_w) == 1
    assert in_w.iloc[0]["device_id"] == 1


@pytest.mark.asyncio
async def test_load_data_mock(tmp_path):
    from hwasync import load_data

    path = tmp_path / "test.xlsx"
    test_df = pd.DataFrame({" Column A ": [1]})
    test_df.to_excel(path, index=False)

    df = await load_data(str(path))
    assert df.columns[0] == "column a"