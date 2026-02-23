import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import statsmodels.api as sm

from q2_config import OUT_DIR
from q2_common import (
    download_ff5_monthly,
    download_spmo_monthly,
    download_umd_factor,
    load_q1_merged,
    merge_on_ym,
)
from q2_config import SPMO_TICKER


def main():
    print("=" * 60)
    print("Q2.4: Fama-French 6-factor controls")
    print("=" * 60)
    loaded = load_q1_merged()
    if loaded is not None:
        _, spmo_returns, df_umd = loaded
        print("Using SPMO and UMD from q2_1_spmo_umd_data.csv")
    else:
        spmo_returns = download_spmo_monthly(ticker=SPMO_TICKER)
        df_umd = download_umd_factor()
    ff5 = download_ff5_monthly()
    ff5_reset = ff5.reset_index()
    ff5_reset["ym"] = ff5_reset["Date"].dt.to_period("M")
    umd_reset = df_umd.reset_index()
    umd_reset["ym"] = umd_reset["Date"].dt.to_period("M")
    ff6 = ff5_reset.merge(umd_reset[["ym", "UMD"]], on="ym", how="inner")
    spmo_df = spmo_returns.reset_index()
    spmo_df.columns = ["Date", "SPMO"]
    spmo_df["ym"] = spmo_df["Date"].dt.to_period("M")
    merge_df = spmo_df.merge(ff6.drop(columns=["Date"]), on="ym", how="inner")
    merge_df["SPMO_excess"] = merge_df["SPMO"] - merge_df["RF"]
    merge_df = merge_df.set_index("Date")
    X_capm = sm.add_constant(merge_df["Mkt-RF"])
    capm = sm.OLS(merge_df["SPMO_excess"], X_capm).fit()
    X_ff6 = sm.add_constant(merge_df[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]])
    ff6_model = sm.OLS(merge_df["SPMO_excess"], X_ff6).fit()
    print("\n" + "=" * 60)
    print("FAMA-FRENCH 6-FACTOR MODEL")
    print("=" * 60)
    print(ff6_model.summary())
    factors = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "UMD"]
    summary_ff6 = pd.DataFrame({
        "Factor": ["Alpha"] + factors,
        "Beta": [ff6_model.params["const"]] + [ff6_model.params[f] for f in factors],
        "T-stat": [ff6_model.tvalues["const"]] + [ff6_model.tvalues[f] for f in factors],
        "P-value": [ff6_model.pvalues["const"]] + [ff6_model.pvalues[f] for f in factors],
    })
    print(summary_ff6.to_string(index=False))
    summary_ff6.to_csv(os.path.join(OUT_DIR, "q2_4_ff6_regression_results.csv"), index=False)
    print("\nSaved: q2_4_ff6_regression_results.csv")
    print("\n--- CAPM vs FF6 market beta ---")
    print(f"  CAPM market beta: {capm.params['Mkt-RF']:.4f}")
    print(f"  FF6 market beta:  {ff6_model.params['Mkt-RF']:.4f}")
    print("\nDone. Outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
