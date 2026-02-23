from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf


def load_fund_monthly_returns(xlsx_path: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns={"return": "fund_ret", "date": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp("M")
    df = df[["date", "fund_ret"]].dropna().sort_values("date").reset_index(drop=True)
    return df


def load_ff5_monthly(parquet_path: Path) -> pd.DataFrame:
    ff = pd.read_parquet(parquet_path)
    ff["dt"] = pd.to_datetime(ff["dt"])
    ff["month"] = ff["dt"].dt.to_period("M").dt.to_timestamp("M")
    factor_cols = ["mkt_rf", "smb", "hml", "rmw", "cma", "rf"]
    for col in factor_cols:
        ff[col] = pd.to_numeric(ff[col], errors="coerce")

    ff_monthly = (
        ff.groupby("month")[factor_cols]
        .apply(lambda x: (1.0 + x).prod() - 1.0)
        .reset_index()
        .rename(columns={"month": "date"})
        .sort_values("date")
        .reset_index(drop=True)
    )
    return ff_monthly


def _fetch_fred_csv(series_id: str) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    out = pd.read_csv(StringIO(r.text))
    out.columns = ["date", series_id]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out[series_id] = pd.to_numeric(out[series_id], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return out


def fetch_external_factors(start: str = "2002-01-01") -> pd.DataFrame:
    # Macro proxies: USD level, 10Y Treasury yield, high-yield OAS, commodity index.
    usd = _fetch_fred_csv("DTWEXBGS")
    dgs10 = _fetch_fred_csv("DGS10")
    hy_oas = _fetch_fred_csv("BAMLH0A0HYM2")

    usd["date"] = usd["date"].dt.to_period("M").dt.to_timestamp("M")
    dgs10["date"] = dgs10["date"].dt.to_period("M").dt.to_timestamp("M")
    hy_oas["date"] = hy_oas["date"].dt.to_period("M").dt.to_timestamp("M")

    usd_monthly = usd.groupby("date", as_index=False)["DTWEXBGS"].last()
    dgs10_monthly = dgs10.groupby("date", as_index=False)["DGS10"].last()
    hy_monthly = hy_oas.groupby("date", as_index=False)["BAMLH0A0HYM2"].last()

    usd_monthly["usd_ret"] = usd_monthly["DTWEXBGS"].pct_change()
    dgs10_monthly["dgs10_chg"] = dgs10_monthly["DGS10"].diff() / 100.0
    hy_monthly["hy_oas_chg"] = hy_monthly["BAMLH0A0HYM2"].diff() / 100.0

    # Commodity proxy from Yahoo; fallback leaves cmdty_ret as missing.
    cmdty = pd.DataFrame(columns=["date", "cmdty_ret"])
    try:
        c = yf.download("^SPGSCI", start=start, auto_adjust=True, progress=False)
        if not c.empty:
            if isinstance(c.columns, pd.MultiIndex):
                close = c["Close"].iloc[:, 0]
            else:
                close = c["Close"]
            close = close.rename("close").to_frame()
            close.index = pd.to_datetime(close.index)
            close["date"] = close.index.to_period("M").to_timestamp("M")
            cmdty = (
                close.groupby("date", as_index=False)["close"].last().assign(
                    cmdty_ret=lambda x: x["close"].pct_change()
                )[["date", "cmdty_ret"]]
            )
    except Exception:
        pass

    out = (
        usd_monthly[["date", "usd_ret"]]
        .merge(dgs10_monthly[["date", "dgs10_chg"]], on="date", how="outer")
        .merge(hy_monthly[["date", "hy_oas_chg"]], on="date", how="outer")
        .merge(cmdty, on="date", how="outer")
        .sort_values("date")
        .reset_index(drop=True)
    )
    out = out[out["date"] >= pd.Timestamp(start)]
    return out


def fetch_hfgm_monthly_returns(start: str = "2022-01-01") -> pd.DataFrame:
    h = yf.download("HFGM", start=start, auto_adjust=True, progress=False)
    if h.empty:
        return pd.DataFrame(columns=["date", "hfgm_ret"])
    if isinstance(h.columns, pd.MultiIndex):
        close = h["Close"].iloc[:, 0]
    else:
        close = h["Close"]

    close = close.rename("close").to_frame()
    close.index = pd.to_datetime(close.index)
    close["date"] = close.index.to_period("M").to_timestamp("M")
    out = (
        close.groupby("date", as_index=False)["close"]
        .last()
        .assign(hfgm_ret=lambda x: x["close"].pct_change())[["date", "hfgm_ret"]]
        .dropna()
        .reset_index(drop=True)
    )
    return out
