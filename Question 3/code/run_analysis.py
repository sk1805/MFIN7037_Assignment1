from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data_prep import (
    fetch_external_factors,
    fetch_hfgm_monthly_returns,
    load_ff5_monthly,
    load_fund_monthly_returns,
)
from model_utils import coef_table, compare_two_models, fit_ols, regression_diagnostics


CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_MD = CODE_DIR / "analysis_global_macro.md"
OUTPUT_DATA = DATA_DIR


def _resolve_input_file(filename: str) -> Path:
    # Prefer files under Question 3/data, while keeping code-dir fallback for compatibility.
    data_path = DATA_DIR / filename
    if data_path.exists():
        return data_path
    return CODE_DIR / filename


def _load_external_factors_csv(csv_path: Path) -> pd.DataFrame:
    ext = pd.read_csv(csv_path)
    if "date" not in ext.columns:
        raise ValueError("external factors CSV is missing 'date' column")
    ext["date"] = pd.to_datetime(ext["date"], errors="coerce")
    ext = ext.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return ext


def to_md_table(df: pd.DataFrame, digits: int = 4) -> str:
    data = df.copy()
    for c in data.columns:
        if pd.api.types.is_numeric_dtype(data[c]):
            data[c] = data[c].map(lambda x: f"{x:.{digits}f}" if pd.notna(x) else "")
    header = "| " + " | ".join(data.columns.astype(str)) + " |"
    sep = "| " + " | ".join(["---"] * len(data.columns)) + " |"
    body = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in data.astype(str).itertuples(index=False, name=None)
    ]
    return "\n".join([header, sep] + body)


def main():
    OUTPUT_DATA.mkdir(exist_ok=True)
    fund_xlsx = _resolve_input_file("CS Global Macro Index at 2x Vol Net of 95bps 2025.09.xlsx")
    ff5_parquet = _resolve_input_file("ff.five_factor.parquet")

    # 1) Load local data
    fund = load_fund_monthly_returns(fund_xlsx)
    ff5 = load_ff5_monthly(ff5_parquet)
    core = fund.merge(ff5, on="date", how="inner").sort_values("date").reset_index(drop=True)
    core["fund_excess"] = core["fund_ret"] - core["rf"]

    # 2) Baseline FF5 model
    ff5_factors = ["mkt_rf", "smb", "hml", "rmw", "cma"]
    ff5_model = fit_ols(core["fund_excess"], core[ff5_factors])
    ff5_diag = regression_diagnostics(ff5_model, core["fund_excess"], core[ff5_factors])
    ff5_coef = coef_table(ff5_model).reset_index().rename(columns={"index": "factor"})

    # 3) External macro factors (with fallback)
    fallback_note = ""
    external_factors_file = OUTPUT_DATA / "external_factors_monthly.csv"
    try:
        ext = fetch_external_factors(start=str(core["date"].min().date()))
        ext.to_csv(external_factors_file, index=False)
    except Exception as e:
        if external_factors_file.exists():
            try:
                ext = _load_external_factors_csv(external_factors_file)
                fallback_note = (
                    f"Online FRED fetch failed ({type(e).__name__}: {e}). "
                    f"Used local fallback file: {external_factors_file.name}."
                )
            except Exception as read_e:
                fallback_note = (
                    f"Online FRED fetch failed ({type(e).__name__}: {e}) and local fallback load failed "
                    f"({type(read_e).__name__}: {read_e}). Macro model uses local-only proxies."
                )
                ext = pd.DataFrame({"date": core["date"]})
        else:
            fallback_note = (
                f"Online FRED fetch failed ({type(e).__name__}: {e}) and no local fallback file was found "
                f"at {external_factors_file}. Macro model uses local-only proxies."
            )
            ext = pd.DataFrame({"date": core["date"]})

    # Local-only proxy candidates if external data unavailable.
    local_proxy = core[["date", "hml", "rmw", "cma"]].copy()
    local_proxy["equity_style_spread"] = core["hml"] + core["rmw"] - core["cma"]

    macro = core.merge(ext, on="date", how="left").merge(local_proxy, on="date", how="left")
    macro["fund_excess"] = macro["fund_ret"] - macro["rf"]

    candidate_order = ["mkt_rf", "usd_ret", "dgs10_chg", "hy_oas_chg", "cmdty_ret", "equity_style_spread"]
    available = [c for c in candidate_order if c in macro.columns and macro[c].notna().sum() > 60]

    # Keep model simple: 3-5 factors (prefer first 5 available).
    if len(available) >= 5:
        macro_factors = available[:5]
    elif len(available) >= 3:
        macro_factors = available[:]
    else:
        macro_factors = ["mkt_rf", "smb", "hml"]

    macro_df = macro[["date", "fund_excess"] + macro_factors].dropna().reset_index(drop=True)
    macro_model = fit_ols(macro_df["fund_excess"], macro_df[macro_factors])
    macro_diag = regression_diagnostics(macro_model, macro_df["fund_excess"], macro_df[macro_factors])
    macro_coef = coef_table(macro_model).reset_index().rename(columns={"index": "factor"})

    # Fair comparison against FF5 over macro sample window.
    same_window = core[core["date"].isin(macro_df["date"])].dropna(subset=["fund_excess"] + ff5_factors)
    ff5_same_model = fit_ols(same_window["fund_excess"], same_window[ff5_factors])
    ff5_same_diag = regression_diagnostics(ff5_same_model, same_window["fund_excess"], same_window[ff5_factors])

    compare_tbl = compare_two_models("FF5 (same window)", ff5_same_diag, "Proposed Macro Model", macro_diag)

    # 4) Extra credit: backtest vs live HFGM
    live_note = ""
    live_stats = pd.DataFrame()
    live_overlap = pd.DataFrame()
    try:
        hfgm = fetch_hfgm_monthly_returns(start="2022-01-01")
        hfgm.to_csv(OUTPUT_DATA / "hfgm_monthly_returns.csv", index=False)
        live = core[["date", "fund_ret"]].merge(hfgm, on="date", how="inner").dropna()
        if len(live) >= 4:
            corr = float(live["fund_ret"].corr(live["hfgm_ret"]))
            beta = float(np.cov(live["hfgm_ret"], live["fund_ret"], ddof=1)[0, 1] / np.var(live["fund_ret"], ddof=1))
            spread = live["hfgm_ret"] - live["fund_ret"]
            te_ann = float(spread.std(ddof=1) * np.sqrt(12.0))
            mean_diff_ann = float(((1.0 + spread.mean()) ** 12) - 1.0)
            live_stats = pd.DataFrame(
                [
                    {
                        "overlap_months": len(live),
                        "corr_hfgm_vs_backtest": corr,
                        "beta_hfgm_on_backtest": beta,
                        "tracking_error_ann": te_ann,
                        "avg_return_diff_ann": mean_diff_ann,
                    }
                ]
            )
            live_overlap = live.copy()
            live_overlap["spread_hfgm_minus_backtest"] = live_overlap["hfgm_ret"] - live_overlap["fund_ret"]
            live_overlap["date"] = live_overlap["date"].dt.strftime("%Y-%m")
        else:
            live_note = "Not enough monthly overlap between HFGM and backtest to estimate robust tracking metrics."
    except Exception as e:
        live_note = f"Could not fetch live HFGM data ({type(e).__name__}: {e})."

    # Save key tables
    ff5_coef.to_csv(CODE_DIR / "ff5_coefficients.csv", index=False)
    macro_coef.to_csv(CODE_DIR / "macro_model_coefficients.csv", index=False)
    compare_tbl.to_csv(CODE_DIR / "model_comparison.csv", index=False)
    if not live_stats.empty:
        live_stats.to_csv(CODE_DIR / "live_vs_backtest_stats.csv", index=False)

    # 5) Build markdown report
    ff5_alpha_p = float(ff5_model.pvalues.get("const", np.nan))
    ff5_alpha_sig = "statistically significant" if ff5_alpha_p < 0.05 else "not statistically significant"
    fit_delta = macro_diag["adj_r2"] - ff5_same_diag["adj_r2"]

    report = f"""# CS Global Macro Index (2x Vol, Net 95bps): Factor Exposure Analysis

## Data Overview

- Fund file: monthly return series from **{core["date"].min().date()}** to **{core["date"].max().date()}** ({len(core)} months).
- FF5 file: daily factor returns compounded to monthly (`mkt_rf`, `smb`, `hml`, `rmw`, `cma`, `rf`).
- Fund excess return is computed as `fund_ret - rf`.

## Is FF5 a Good Benchmark Conceptually?

FF5 is partially useful but incomplete for a global macro product:

- **Reasonable for** investors who mainly care whether the fund is just repackaged equity style risk (market, size, value, profitability, investment).
- **Inappropriate for** investors expecting a true global macro profile, because macro funds typically load on rates, FX, commodities, and credit spread dynamics beyond equity cross-section factors.

So FF5 is a good **diagnostic benchmark** (to test equity-style dependence), but not a complete **economic benchmark** for strategy intent.

## FF5 Regression Results (Main Question)

### Model fit and alpha

{to_md_table(pd.DataFrame([ff5_diag]))}

Alpha under FF5 is **{ff5_alpha_sig}** at the 5% level (p-value = {ff5_alpha_p:.4f}).

### FF5 exposures

{to_md_table(ff5_coef)}

Interpretation (best effort from signs/magnitudes):
- The fund has some equity beta, but FF5 explainability is limited (adj. R² is modest).
- Residual risk remains large, suggesting exposures outside standard equity style factors.

## Proposed 3-5 Factor Macro Model (No Circular Global-Macro Index)

Chosen factors: **{", ".join(macro_factors)}**

Rationale:
- `mkt_rf`: broad equity risk premium.
- `usd_ret`: broad USD move proxy for macro FX exposure.
- `dgs10_chg`: global duration/rates shock proxy (US 10Y yield change).
- `hy_oas_chg`: credit risk appetite / stress proxy.
- `cmdty_ret`: commodity risk proxy.
- `equity_style_spread` (fallback/local): additional equity-style cycle proxy when external series are constrained.

### Macro model exposures

{to_md_table(macro_coef)}

### Explainability vs FF5

{to_md_table(compare_tbl)}

Improvement in adjusted R² (macro model minus FF5 on same window): **{fit_delta:.4f}**.

### Do benchmark covariances make sense for global macro?

Yes, more than FF5 alone. A global macro process should co-move with:
- rates repricing,
- USD cycles,
- credit stress/risk-on regimes,
- commodity regimes,
in addition to equity beta.

This covariance structure is closer to the strategy narrative of delivering hedge-fund-like multi-asset premia.

## Risk Premia Interpretation

Findings are broadly consistent with a **risk premia harvesting** view if macro-factor loadings are stable/significant:
- Time-varying compensation for macro risks can explain part of returns.
- Any remaining alpha may reflect dynamic timing, implementation edge, or model omission.

Supporting references:
- Fama, E. F., and French, K. R. (2015), *A five-factor asset pricing model*.
- Asness, C. S., Moskowitz, T. J., and Pedersen, L. H. (2013), *Value and Momentum Everywhere*.
- Koijen, R. S. J., Moskowitz, T. J., Pedersen, L. H., and Vrugt, E. B. (2018), *Carry*.
- Moskowitz, T. J., Ooi, Y. H., and Pedersen, L. H. (2012), *Time series momentum*.

## Extra Credit: Backtest vs Live (HFGM)

"""

    if not live_stats.empty:
        report += to_md_table(live_stats) + "\n\n"
        report += "Exact overlap months and returns used in the calculation:\n\n"
        report += to_md_table(live_overlap[["date", "fund_ret", "hfgm_ret", "spread_hfgm_minus_backtest"]]) + "\n\n"
        report += "The backtest and live ETF are directionally related, but this estimate is based on a short overlap sample and should be treated as preliminary.\n"
    else:
        report += f"- {live_note or 'Live comparison unavailable.'}\n"

    if fallback_note:
        report += f"\n## Data Constraint Note\n\n- {fallback_note}\n"

    report += """
## Bottom Line

- FF5 alone is **not** a fully appropriate benchmark for this fund's macro mandate.
- FF5 is still useful as an equity-risk sanity check (alpha/exposure diagnostic).
- A mixed macro benchmark with equities + rates + FX + credit + commodities is more economically aligned and generally improves explainability.
"""

    OUTPUT_MD.write_text(report, encoding="utf-8")
    print(f"Analysis complete. Report saved to: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
