import os
from io import BytesIO, StringIO
import zipfile

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from q2_config import (
    END_DATE,
    OUT_DIR,
    REQUEST_TIMEOUT,
    RETURN_MAX,
    RETURN_MIN,
    START_DATE,
    URL_DECILES,
    URL_FF5,
    URL_UMD,
)


def download_spmo_monthly(ticker="SPMO", start=None, end=None):
    """Download ETF monthly returns (default SPMO). Returns series with DatetimeIndex."""
    start = start or START_DATE
    end = end or END_DATE
    print(f"Downloading {ticker} data...")
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if data.empty:
        data = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    close = (
        data["Adj Close"]
        if "Adj Close" in data.columns
        else data["Close"]
        if "Close" in data.columns
        else data.iloc[:, -1]
    )
    monthly = close.resample("ME").last().dropna()
    ret = monthly.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    ret = ret[(ret >= RETURN_MIN) & (ret <= RETURN_MAX)]
    ret.name = ticker
    print(f"  {ticker}: {ret.index.min().strftime('%Y-%m')} to {ret.index.max().strftime('%Y-%m')}, n={len(ret)}")
    return ret


def download_umd_factor(url=None):
    """Download Fama-French momentum factor (UMD). Returns DataFrame with UMD column, month-end index."""
    url = url or URL_UMD
    print("Downloading Fama-French Momentum Factor...")
    r = requests.get(url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    with zipfile.ZipFile(BytesIO(r.content)) as z:
        raw = z.read(z.namelist()[0]).decode("utf-8", errors="replace")
    lines = raw.split("\n")
    start = 0
    for i, line in enumerate(lines):
        parts = line.split(",")
        if parts and len(parts[0].strip()) == 6 and parts[0].strip().isdigit():
            start = i
            break
    df = pd.read_csv(StringIO("\n".join(lines[start:])), header=None, names=["Date", "Mom"])
    df = df[df["Date"].notna()].copy()
    df = df[df["Date"].astype(str).str.strip().str.len() == 6].copy()
    df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m")
    df["UMD"] = pd.to_numeric(df["Mom"], errors="coerce") / 100
    df = df.set_index("Date")[["UMD"]].dropna()
    df.index = df.index.to_period("M").to_timestamp(how="end").normalize()
    df = df[~df["UMD"].isna()].dropna()
    print(f"  UMD: {df.index.min().strftime('%Y-%m')} to {df.index.max().strftime('%Y-%m')}, n={len(df)}")
    return df


def download_ff5_monthly(url=None):
    """Download Fama-French 5 factors (monthly). Returns DataFrame with Mkt-RF, SMB, HML, RMW, CMA, RF."""
    url = url or URL_FF5
    print("Downloading Fama-French 5 factors...")
    r = requests.get(url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    with zipfile.ZipFile(BytesIO(r.content)) as z:
        raw = z.read(z.namelist()[0]).decode("utf-8", errors="replace")
    lines = raw.split("\n")
    start = 0
    for i, line in enumerate(lines):
        parts = line.split(",")
        if parts and len(parts[0].strip()) == 6 and parts[0].strip().isdigit():
            start = i
            break
    df = pd.read_csv(
        StringIO("\n".join(lines[start:])),
        header=None,
        names=["Date", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"],
    )
    df = df[df["Date"].notna()].copy()
    df = df[df["Date"].astype(str).str.strip().str.len() == 6].copy()
    df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m")
    df = df.set_index("Date")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce") / 100
    df.index = df.index.to_period("M").to_timestamp(how="end").normalize()
    df = df.dropna()
    print(f"  FF5: {df.index.min().strftime('%Y-%m')} to {df.index.max().strftime('%Y-%m')}")
    return df


def download_momentum_deciles(url=None):
    """Download 10 portfolios (Prior 12-2) from Ken French. Returns DataFrame with VW_D1..VW_D10, EW_D1..EW_D10."""
    url = url or URL_DECILES
    print("Downloading momentum decile portfolios (Prior 12-2)...")
    r = requests.get(url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    with zipfile.ZipFile(BytesIO(r.content)) as z:
        raw = z.read(z.namelist()[0]).decode("utf-8", errors="replace")
    lines = raw.split("\n")
    vw_start = ew_start = None
    for i, line in enumerate(lines):
        if "Value Weight" in line and "Returns" in line and "Monthly" in line:
            vw_start = i + 1
        if "Equal Weight" in line and "Returns" in line and "Monthly" in line:
            ew_start = i + 1
    if vw_start is None or ew_start is None:
        for i, line in enumerate(lines):
            p = line.split(",")
            if p and len(p[0].strip()) == 6 and p[0].strip().isdigit():
                vw_start = i
                break
        monthly_lines = [l for l in lines[vw_start:] if l.strip() and len(l.split(",")[0].strip()) == 6]
        arr = []
        for l in monthly_lines:
            parts = [x.strip() for x in l.split(",")]
            if len(parts) >= 20:
                arr.append(parts[:20])
            elif len(parts) >= 10:
                arr.append(parts[:10] + [np.nan] * 10)
        if not arr:
            raise ValueError("Could not parse decile file")
        df = pd.DataFrame(arr)
        df.columns = ["Date"] + [f"VW_D{i}" for i in range(1, 11)] + [f"EW_D{i}" for i in range(1, 11)]
        df["Date"] = pd.to_datetime(df["Date"], format="%Y%m")
        for c in df.columns:
            if c != "Date":
                df[c] = pd.to_numeric(df[c], errors="coerce") / 100
        df = df.set_index("Date").resample("ME").last()
        print(f"  Deciles: {df.index.min().strftime('%Y-%m')} to {df.index.max().strftime('%Y-%m')}")
        return df

    def parse_section(from_line, ncols=10, prefix=""):
        rows = []
        for line in lines[from_line:]:
            parts = [x.strip() for x in line.split(",")]
            if not parts or len(parts[0]) != 6 or not parts[0].isdigit():
                if rows:
                    break
                continue
            if len(parts) >= ncols + 1:
                rows.append(parts[: ncols + 1])
        if not rows:
            return None
        d = pd.DataFrame(rows)
        d.columns = ["Date"] + [f"{prefix}D{i}" for i in range(1, ncols + 1)]
        d["Date"] = pd.to_datetime(d["Date"], format="%Y%m")
        for c in d.columns:
            if c != "Date":
                d[c] = pd.to_numeric(d[c], errors="coerce") / 100
        return d.set_index("Date")

    vw_df = parse_section(vw_start, 10, "VW_")
    ew_df = parse_section(ew_start, 10, "EW_")
    if vw_df is not None and ew_df is not None:
        df = vw_df.join(ew_df)
    else:
        monthly_lines = [l for l in lines[12:] if l.strip()]
        parsed = []
        for l in monthly_lines:
            p = [x.strip() for x in l.split(",")]
            if len(p) >= 21 and len(p[0]) == 6 and p[0].isdigit():
                parsed.append(p[:21])
        df = pd.DataFrame(parsed)
        df.columns = ["Date"] + [f"VW_D{i}" for i in range(1, 11)] + [f"EW_D{i}" for i in range(1, 11)]
        df["Date"] = pd.to_datetime(df["Date"], format="%Y%m")
        for c in df.columns:
            if c != "Date":
                df[c] = pd.to_numeric(df[c], errors="coerce") / 100
        df = df.set_index("Date")
    df = df.resample("ME").last()
    print(f"  Deciles: {df.index.min().strftime('%Y-%m')} to {df.index.max().strftime('%Y-%m')}")
    return df


def merge_on_ym(left_series, right_df, left_name="left"):
    """Align left_series (Series with DatetimeIndex) and right_df (DataFrame with Date index) by year-month. Returns DataFrame with left_name column and right_df columns."""
    if left_series.index.tz is not None:
        left_series = left_series.tz_localize(None)
    left_df = left_series.reset_index()
    left_df.columns = ["Date", left_name]
    left_df["ym"] = left_df["Date"].dt.to_period("M")
    right_reset = right_df.reset_index()
    right_reset["ym"] = right_reset["Date"].dt.to_period("M")
    merged = left_df.merge(right_reset.drop(columns=["Date"]), on="ym", how="inner")
    return merged.set_index("Date")


def load_q1_merged(path=None):
    """Load merged SPMO-UMD data from Q2.1 output if it exists. Returns (df_merged, spmo_returns, df_umd) or None."""
    path = path or os.path.join(OUT_DIR, "q2_1_spmo_umd_data.csv")
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    spmo = df["SPMO"]
    umd = df[["UMD"]]
    return df, spmo, umd
