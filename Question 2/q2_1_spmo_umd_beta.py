
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import jarque_bera, probplot
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

from q2_config import OUT_DIR, SPMO_TICKER
from q2_common import (
    download_ff5_monthly,
    download_spmo_monthly,
    download_umd_factor,
    merge_on_ym,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

os.environ.setdefault("MPLCONFIGDIR", OUT_DIR)


def main():
    print("=" * 60)
    print("Q2.1: SPMO beta to UMD factor")
    print("=" * 60)
    spmo_returns = download_spmo_monthly(ticker=SPMO_TICKER)
    df_umd = download_umd_factor()
    ff5 = download_ff5_monthly()
    # Merge SPMO with UMD on year-month
    df_merged = merge_on_ym(spmo_returns, df_umd, left_name="SPMO")[["SPMO", "UMD"]].dropna()
    # Merge with FF5 to get Mkt-RF and RF for market-controlled regression
    ff5_reset = ff5.reset_index()
    ff5_reset["ym"] = ff5_reset["Date"].dt.to_period("M")
    umd_reset = df_umd.reset_index()
    umd_reset["ym"] = umd_reset["Date"].dt.to_period("M")
    ff6 = ff5_reset.merge(umd_reset[["ym", "UMD"]], on="ym", how="inner")
    spmo_df = spmo_returns.reset_index()
    spmo_df.columns = ["Date", "SPMO"]
    spmo_df["ym"] = spmo_df["Date"].dt.to_period("M")
    df_merged = spmo_df.merge(ff6.drop(columns=["Date"]), on="ym", how="inner")
    df_merged = df_merged.set_index("Date")[["SPMO", "UMD", "Mkt-RF", "RF"]].dropna()
    df_merged["SPMO_excess"] = df_merged["SPMO"] - df_merged["RF"]

    # --- Debug: alignment and summary stats ---
    print("\n--- Merge debug: SPMO vs UMD ---")
    print("SPMO  mean = {:.6f}, std = {:.6f}".format(df_merged["SPMO"].mean(), df_merged["SPMO"].std()))
    print("UMD   mean = {:.6f}, std = {:.6f}".format(df_merged["UMD"].mean(), df_merged["UMD"].std()))
    print("Correlation(SPMO, UMD) = {:.4f}".format(df_merged["SPMO"].corr(df_merged["UMD"])))
    print("First 5 rows of merged data:")
    print(df_merged[["SPMO", "UMD"]].head().to_string())
    print("\nLag correlations (SPMO vs UMD):")
    for lag in (-1, 0, 1):
        if lag == 0:
            c = df_merged["SPMO"].corr(df_merged["UMD"])
        elif lag == -1:
            c = df_merged["SPMO"].shift(-1).corr(df_merged["UMD"])
        else:
            c = df_merged["SPMO"].corr(df_merged["UMD"].shift(-1))
        print("  Lag {:+.0f}: {:.4f}".format(lag, c))
    print("\nMerged: {} months, {} to {}".format(
        len(df_merged), df_merged.index.min().strftime("%Y-%m"), df_merged.index.max().strftime("%Y-%m")))

    # (1) Simple regression: SPMO ~ UMD (for reference; biased by omitted market)
    X_simple = sm.add_constant(df_merged["UMD"])
    model_simple = sm.OLS(df_merged["SPMO"], X_simple).fit()
    print("\n" + "=" * 60)
    print("(1) SIMPLE: SPMO = α + β(UMD) + ε  [omitted market bias]")
    print("=" * 60)
    print(model_simple.summary())
    print("  Beta(UMD) = {:.4f}, R² = {:.4f}".format(model_simple.params["UMD"], model_simple.rsquared))

    # (2) Market-controlled: SPMO_excess ~ Mkt-RF + UMD (economically meaningful UMD beta)
    X_ff2 = sm.add_constant(df_merged[["Mkt-RF", "UMD"]])
    model = sm.OLS(df_merged["SPMO_excess"], X_ff2).fit()
    alpha = model.params["const"]
    beta_umd = model.params["UMD"]
    r2 = model.rsquared
    residuals = model.resid
    fitted = model.fittedvalues

    print("\n" + "=" * 60)
    print("(2) MARKET-CONTROLLED: SPMO_excess = α + β_mkt(Mkt-RF) + β_umd(UMD) + ε")
    print("=" * 60)
    print(model.summary())
    print("\n--- Key results (use market-controlled for report) ---")
    print("Beta (UMD, controlling for market): {:.4f}  t={:.2f}  p={:.4f}".format(
        beta_umd, model.tvalues["UMD"], model.pvalues["UMD"]))
    print("R²: {:.4f}".format(r2))
    alpha_ann = (1 + alpha) ** 12 - 1
    print("Alpha (monthly): {:.6f}  ({:.2%} annualized)".format(alpha, alpha_ann))

    jb_stat, jb_p = jarque_bera(residuals)
    bp_stat, bp_p, _, _ = het_breuschpagan(residuals, X_ff2)
    dw = durbin_watson(residuals)
    print("\n--- Diagnostics (market-controlled model) ---")
    print("Jarque-Bera: {:.2f} (p={:.4f}), Breusch-Pagan: {:.2f} (p={:.4f}), Durbin-Watson: {:.2f}".format(
        jb_stat, jb_p, bp_stat, bp_p, dw))

    print("\n--- Variance decomposition (market-controlled) ---")
    print("Explained by Mkt-RF + UMD: {:.1f}%, Unexplained: {:.1f}%".format(r2 * 100, (1 - r2) * 100))

    alpha_ann = (1 + alpha) ** 12 - 1
    summary = pd.DataFrame({
        "Metric": [
            "Beta (UMD)", "Alpha (monthly)", "Alpha (annualized)",
            "Alpha t-stat", "Beta t-stat", "R-squared", "Adj R-squared",
            "Correlation(SPMO, UMD)", "Residual Std (monthly)", "N", "Start", "End",
            "Beta (UMD) simple", "R-squared simple",
        ],
        "Value": [
            "{:.4f}".format(beta_umd), "{:.6f}".format(alpha), "{:.4f}".format(alpha_ann),
            "{:.2f}".format(model.tvalues["const"]), "{:.2f}".format(model.tvalues["UMD"]),
            "{:.4f}".format(r2), "{:.4f}".format(model.rsquared_adj),
            "{:.4f}".format(df_merged["SPMO"].corr(df_merged["UMD"])),
            "{:.4f}".format(np.sqrt(model.mse_resid)), str(len(df_merged)),
            df_merged.index.min().strftime("%Y-%m"), df_merged.index.max().strftime("%Y-%m"),
            "{:.4f}".format(model_simple.params["UMD"]), "{:.4f}".format(model_simple.rsquared),
        ],
    })
    summary.to_csv(os.path.join(OUT_DIR, "q2_1_regression_summary.csv"), index=False)
    df_merged[["SPMO", "UMD"]].to_csv(os.path.join(OUT_DIR, "q2_1_spmo_umd_data.csv"))
    print("\nSaved: q2_1_regression_summary.csv, q2_1_spmo_umd_data.csv")

    if HAS_MPL and np.isfinite(r2) and np.isfinite(residuals).all():
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        ax1, ax2, ax3, ax4 = axes.flat
        ax1.scatter(df_merged["UMD"] * 100, df_merged["SPMO_excess"] * 100, alpha=0.6, s=25)
        ax1.plot(df_merged["UMD"] * 100, (model.params["const"] + model.params["UMD"] * df_merged["UMD"]) * 100, "r-", lw=2,
                 label="β_UMD={:.3f} (ctrl Mkt), R²={:.3f}".format(beta_umd, r2))
        ax1.set_xlabel("UMD (%)"); ax1.set_ylabel("SPMO excess (%)"); ax1.legend(); ax1.grid(True, alpha=0.3)
        ax2.plot(df_merged.index, residuals * 100, "o-", ms=2, alpha=0.7)
        ax2.axhline(0, color="red", ls="--"); ax2.set_xlabel("Date"); ax2.set_ylabel("Residual (%)"); ax2.grid(True, alpha=0.3)
        ax3.hist(residuals * 100, bins=25, density=True, alpha=0.7, edgecolor="k")
        ax3.set_xlabel("Residual (%)"); ax3.set_ylabel("Density"); ax3.grid(True, alpha=0.3)
        probplot(residuals, dist="norm", plot=ax4); ax4.set_title("Q-Q"); ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "q2_1_spmo_umd_regression_diagnostics.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved: q2_1_spmo_umd_regression_diagnostics.png")
    print("\nDone. Outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
