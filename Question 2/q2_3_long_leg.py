import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import statsmodels.api as sm

from q2_config import OUT_DIR
from q2_common import (
    download_momentum_deciles,
    download_spmo_monthly,
    download_umd_factor,
    load_q1_merged,
    merge_on_ym,
)
from q2_config import SPMO_TICKER

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

os.environ.setdefault("MPLCONFIGDIR", OUT_DIR)


def _decile_sort(cols):
    def key(c):
        for i in range(1, 11):
            if f"D{i}" in c or c.endswith(str(i)):
                return i
        return 0
    return sorted([c for c in cols if key(c) > 0], key=key)


def main():
    print("=" * 60)
    print("Q2.3: Beta to long-leg; VW vs EW momentum")
    print("=" * 60)
    loaded = load_q1_merged()
    if loaded is not None:
        _, spmo_returns, df_umd = loaded
        print("Using SPMO and UMD from q2_1_spmo_umd_data.csv")
    else:
        spmo_returns = download_spmo_monthly(ticker=SPMO_TICKER)
        df_umd = download_umd_factor()
    decile_data = download_momentum_deciles()
    vw_cols = _decile_sort([c for c in decile_data.columns if "VW" in c or "VW_" in str(c)])
    ew_cols = _decile_sort([c for c in decile_data.columns if "EW" in c or "EW_" in str(c)])
    if not vw_cols:
        vw_cols = list(decile_data.columns)[:10]
    if not ew_cols:
        ew_cols = list(decile_data.columns)[-10:] if len(decile_data.columns) >= 20 else vw_cols
    winners_vw = decile_data[vw_cols[-1]]
    losers_vw = decile_data[vw_cols[0]]
    winners_ew = decile_data[ew_cols[-1]] if ew_cols else winners_vw
    losers_ew = decile_data[ew_cols[0]] if ew_cols else losers_vw
    mom_ls_vw = winners_vw - losers_vw
    mom_ls_ew = winners_ew - losers_ew
    spmo_mom = pd.DataFrame({
        "SPMO": spmo_returns,
        "Winners_VW": winners_vw, "Winners_EW": winners_ew,
        "MomLS_VW": mom_ls_vw, "MomLS_EW": mom_ls_ew,
        "UMD_Official": df_umd["UMD"],
    }).dropna()
    models = {}
    for name, xcol in [
        ("Winners_VW", "Winners_VW"), ("Winners_EW", "Winners_EW"),
        ("UMD_Official", "UMD_Official"), ("MomLS_VW", "MomLS_VW"), ("MomLS_EW", "MomLS_EW"),
    ]:
        X = sm.add_constant(spmo_mom[xcol])
        models[name] = sm.OLS(spmo_mom["SPMO"], X).fit()
    comp = pd.DataFrame({
        "Model": list(models.keys()),
        "Beta": [models[m].params.iloc[1] for m in models],
        "T-stat": [models[m].tvalues.iloc[1] for m in models],
        "R-squared": [models[m].rsquared for m in models],
        "Alpha (annual %)": [models[m].params["const"] * 12 * 100 for m in models],
    })
    print("\n" + "=" * 60)
    print("SPMO vs LONG LEG and LONG-SHORT")
    print("=" * 60)
    print(comp.to_string(index=False))
    valid_r2 = comp["R-squared"].replace([np.nan, -np.inf], np.nan).dropna()
    best = comp.loc[valid_r2.idxmax(), "Model"] if len(valid_r2) else comp["Model"].iloc[0]
    print(f"\nBest fit (highest R²): {best}")
    cor_vw = spmo_mom["MomLS_VW"].corr(spmo_mom["UMD_Official"])
    cor_ew = spmo_mom["MomLS_EW"].corr(spmo_mom["UMD_Official"])
    print(f"Correlation with official UMD: MomLS_VW={cor_vw:.4f}, MomLS_EW={cor_ew:.4f}")
    comp.to_csv(os.path.join(OUT_DIR, "q2_3_all_models_summary.csv"), index=False)
    pd.DataFrame({
        "Winners_VW": winners_vw, "Winners_EW": winners_ew,
        "Losers_VW": losers_vw, "Losers_EW": losers_ew,
        "MomLS_VW": mom_ls_vw, "MomLS_EW": mom_ls_ew,
        "UMD_Official": df_umd["UMD"],
    }).dropna().to_csv(os.path.join(OUT_DIR, "q2_3_momentum_portfolios.csv"))
    print("Saved: q2_3_all_models_summary.csv, q2_3_momentum_portfolios.csv")
    if HAS_MPL:
        order = ["Winners_VW", "Winners_EW", "UMD_Official", "MomLS_VW", "MomLS_EW"]
        comp_plot = comp.set_index("Model").loc[order].reset_index()
        betas = comp_plot["Beta"]
        r2s = comp_plot["R-squared"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        colors = ["#2ecc71" if b > 0 else "#e74c3c" for b in betas]
        axes[0].barh(comp_plot["Model"], betas, alpha=0.7, color=colors)
        axes[0].set_xlabel("Beta"); axes[0].set_title("SPMO beta to momentum portfolios")
        axes[0].axvline(0, color="black", linewidth=0.8)
        b_max = max(abs(betas.max()), abs(betas.min()), 0.1)
        axes[0].set_xlim(-b_max - 0.05, b_max + 0.05)
        axes[0].grid(True, alpha=0.3, axis="x")
        axes[1].barh(comp_plot["Model"], r2s, alpha=0.7, color="steelblue")
        axes[1].set_xlabel("R²"); axes[1].set_title("R-squared"); axes[1].set_xlim(0, min(1.0, max(r2s) * 1.2 + 0.05))
        axes[1].grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "q2_3_momentum_decomposition.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved: q2_3_momentum_decomposition.png")
    print("\nDone. Outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
