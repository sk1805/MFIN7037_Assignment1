import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import statsmodels.api as sm

from q2_config import OUT_DIR, OTHER_ETF_TICKERS
from q2_common import (
    download_ff5_monthly,
    download_spmo_monthly,
    download_umd_factor,
    load_q1_merged,
)


def main():
    print("=" * 60)
    print("Q2.5: Other momentum ETFs – FF6 loadings")
    print("=" * 60)
    loaded = load_q1_merged()
    if loaded is not None:
        _, _, df_umd = loaded
        print("Using UMD from q2_1_spmo_umd_data.csv")
    else:
        df_umd = download_umd_factor()
    ff5 = download_ff5_monthly()
    ff5_reset = ff5.reset_index()
    ff5_reset["ym"] = ff5_reset["Date"].dt.to_period("M")
    umd_reset = df_umd.reset_index()
    umd_reset["ym"] = umd_reset["Date"].dt.to_period("M")
    ff6 = ff5_reset.merge(umd_reset[["ym", "UMD"]], on="ym", how="inner")
    results = []
    for ticker, name in OTHER_ETF_TICKERS:
        print(f"\nDownloading {ticker} ({name})...")
        try:
            ret = download_spmo_monthly(ticker=ticker)
            ret.name = ticker
        except Exception as e:
            print(f"  Skip {ticker}: {e}")
            results.append({"ticker": ticker, "name": name, "error": str(e)})
            continue
        ret_df = ret.reset_index()
        ret_df.columns = ["Date", ticker]
        ret_df["ym"] = ret_df["Date"].dt.to_period("M")
        merge_df = ret_df.merge(ff6, on="ym", how="inner")
        merge_df[f"{ticker}_excess"] = merge_df[ticker] - merge_df["RF"]
        X = sm.add_constant(merge_df[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]])
        model = sm.OLS(merge_df[f"{ticker}_excess"], X).fit()
        results.append({
            "ticker": ticker, "name": name,
            "alpha_ann": ((1 + model.params["const"]) ** 12 - 1) * 100,
            "Mkt-RF": model.params["Mkt-RF"], "SMB": model.params["SMB"],
            "HML": model.params["HML"], "RMW": model.params["RMW"], "CMA": model.params["CMA"],
            "UMD": model.params["UMD"], "R2": model.rsquared, "nobs": int(model.nobs),
        })
        print(f"  {ticker} FF6: Mkt-RF={model.params['Mkt-RF']:.3f}, SMB={model.params['SMB']:.3f}, UMD={model.params['UMD']:.3f}, R2={model.rsquared:.3f}")
    ok = [r for r in results if "error" not in r]
    if ok:
        pd.DataFrame([{k: v for k, v in r.items() if k != "name"} for r in ok]).to_csv(
            os.path.join(OUT_DIR, "q2_5_other_etfs_ff6.csv"), index=False
        )
    print("\nSaved: q2_5_other_etfs_ff6.csv")
    print("\nDone. Outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
