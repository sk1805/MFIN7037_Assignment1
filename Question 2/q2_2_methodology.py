import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd

from q2_config import (
    COMPARISON_FEATURES,
    COMPARISON_SPMO,
    COMPARISON_UMD,
    OUT_DIR,
    SPMO_METHODOLOGY,
    UMD_METHODOLOGY,
)


def main():
    print("=" * 60)
    print("Q2.2: SPMO methodology vs UMD construction")
    print("=" * 60)
    print("\nSPMO METHODOLOGY (official sources)")
    print("-" * 50)
    for k, v in SPMO_METHODOLOGY.items():
        print(f"  {k:<22}: {v}")
    print("\nACADEMIC UMD METHODOLOGY (Fama-French)")
    print("-" * 50)
    for k, v in UMD_METHODOLOGY.items():
        print(f"  {k:<22}: {v}")

    comparison = pd.DataFrame({
        "Feature": COMPARISON_FEATURES,
        "UMD (Fama-French)": COMPARISON_UMD,
        "SPMO": COMPARISON_SPMO,
    })
    print("\n--- Comparison table ---")
    print(comparison.to_string(index=False))
    comparison.to_csv(os.path.join(OUT_DIR, "q2_2_methodology_comparison.csv"), index=False)
    print("\nSaved: q2_2_methodology_comparison.csv")

    summary_path = os.path.join(OUT_DIR, "q2_1_regression_summary.csv")
    if os.path.isfile(summary_path):
        s = pd.read_csv(summary_path)
        beta_row = s[s["Metric"] == "Beta (UMD)"]
        if not beta_row.empty:
            beta_val = float(beta_row["Value"].iloc[0])
            print("\n--- Observed beta (from Q2.1) ---")
            print(f"  Predicted β ≈ 0.20–0.30  |  Observed β = {beta_val:.2f}")
    print("\nDone. Outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
