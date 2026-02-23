# Q2: Smart Beta ETFs (SPMO)

## Layout

- **Config:** `q2_config.py` – dates, tickers, URLs, methodology text. Edit here to change sample period or other ETFs.
- **Shared helpers:** `q2_common.py` – downloads (SPMO, UMD, FF5, deciles), merge by year-month.
- **One script per question:**
  - `q2_1_spmo_umd_beta.py` – Q2.1: Beta to UMD; is ETF broken?
  - `q2_2_methodology.py` – Q2.2: SPMO quote and vs UMD construction.
  - `q2_3_long_leg.py` – Q2.3: Beta to long-leg; long/short; VW vs EW.
  - `q2_4_ff6_controls.py` – Q2.4: FF6 controls; market beta, size bias.
  - `q2_5_other_etfs.py` – Q2.5: Two other momentum ETFs, FF6 loadings.
- **Report:** `q2_report.py` – reads all `q2_*` CSVs and writes **REPORT_Q2.md** and **REPORT_Q2.pdf**.
- **Run all:** `q2_run_all.py` – runs 1 → 2 → 3 → 4 → 5 → report.

## How to run

From this **Q2** folder (with your Python env/venv already activated):

```bash
pip install -r requirements.txt   # includes yfinance, requests, statsmodels, scipy, matplotlib, reportlab
```

**Run everything (recommended):**

```bash
python q2_run_all.py
```

**Run one question at a time:**

```bash
python q2_1_spmo_umd_beta.py
python q2_2_methodology.py
python q2_3_long_leg.py
python q2_4_ff6_controls.py
python q2_5_other_etfs.py
python q2_report.py
```

## Outputs (in this `Q2` folder)

| File | From |
|------|------|
| `q2_1_regression_summary.csv`, `q2_1_spmo_umd_data.csv`, `q2_1_...diagnostics.png` | q2_1 |
| `q2_2_methodology_comparison.csv` | q2_2 |
| `q2_3_all_models_summary.csv`, `q2_3_momentum_portfolios.csv`, `q2_3_...decomposition.png` | q2_3 |
| `q2_4_ff6_regression_results.csv` | q2_4 |
| `q2_5_other_etfs_ff6.csv` | q2_5 |
| **REPORT_Q2.md** | q2_report |
| **REPORT_Q2.pdf** | q2_report (requires `reportlab`) |

## About

This `Q2` package implements the Smart Beta ETF analysis for MFIN 7037 Question 2. Configuration (dates, tickers, data URLs, methodology text) lives in `q2_config.py`, shared data helpers in `q2_common.py`, each question has its own `q2_X_*.py` script, and `q2_report.py` turns the CSV outputs into a Markdown + PDF report.
