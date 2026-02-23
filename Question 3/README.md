# Global Macro Factor Attribution (Assignment 1)

This project analyzes the `CS Global Macro Index at 2x Vol Net of 95bps` series and answers:

- whether FF5 is a reasonable benchmark,
- FF5 alpha and exposures,
- a better 3-5 factor macro benchmark,
- improvement in explainability vs FF5,
- extra-credit backtest vs live ETF (`HFGM`) tracking check.

## Project Files

- `run_analysis.py`  
  Main entry point. Runs the full pipeline and writes outputs.

- `data_prep.py`  
  Loads local data (Excel + parquet), converts FF5 daily -> monthly, fetches external macro series and `HFGM` monthly returns.

- `model_utils.py`  
  OLS helpers, diagnostics, coefficient tables, model comparison utilities.

## Inputs

Place these in the `data/` folder:

- `CS Global Macro Index at 2x Vol Net of 95bps 2025.09.xlsx`
- `ff.five_factor.parquet`

## Outputs

Generated after running:

- `analysis_global_macro.md` (main write-up)
- `ff5_coefficients.csv`
- `macro_model_coefficients.csv`
- `model_comparison.csv`
- `live_vs_backtest_stats.csv` (when overlap exists)
- `data/external_factors_monthly.csv`
- `data/hfgm_monthly_returns.csv`

## How To Run

From the `Question 3` folder:

1. Create and activate a virtual environment
   - macOS/Linux:
     - `python3 -m venv .venv`
     - `source .venv/bin/activate`

2. Install dependencies
   - `pip install -r requirements.txt`

3. Run analysis
   - `python code/run_analysis.py`

## Notes

- The script uses online data sources (FRED and Yahoo Finance) for macro proxies and `HFGM`.
- If online access fails, the code falls back to local-only behavior where possible and reports constraints in the markdown output.
