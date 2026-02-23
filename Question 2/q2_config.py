import os
from datetime import datetime

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
Q2_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = Q2_DIR
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Sample period (used for SPMO and other ETFs)
# ---------------------------------------------------------------------------
START_DATE = "2015-10-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")  # today, or set e.g. "2025-12-31"

# ---------------------------------------------------------------------------
# Tickers
# ---------------------------------------------------------------------------
SPMO_TICKER = "SPMO"

# Other momentum ETFs for Q2.5: list of (ticker, full_name)
OTHER_ETF_TICKERS = [
    ("MTUM", "iShares MSCI USA Momentum Factor ETF"),
    ("QMOM", "Alpha Architect US Quantitative Momentum ETF"),
]

# ---------------------------------------------------------------------------
# Ken French data URLs
# ---------------------------------------------------------------------------
URL_UMD = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip"
URL_FF5 = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
URL_DECILES = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/10_Portfolios_Prior_12_2_CSV.zip"

# ---------------------------------------------------------------------------
# Request timeout (seconds)
# ---------------------------------------------------------------------------
REQUEST_TIMEOUT = 30

# ---------------------------------------------------------------------------
# Data cleaning: monthly return bounds (drop if outside)
# ---------------------------------------------------------------------------
RETURN_MIN = -0.5
RETURN_MAX = 0.5

# ---------------------------------------------------------------------------
# SPMO methodology (from Invesco/S&P; verify against latest prospectus)
# ---------------------------------------------------------------------------
SPMO_METHODOLOGY = {
    "Source": "Invesco SPMO Prospectus / S&P 500 Momentum Index methodology",
    "Index Name": "S&P 500 Momentum Index",
    "Selection Criteria": "Highest momentum score from S&P 500 constituents",
    "Number of Holdings": "Approximately 100 stocks",
    "Momentum Definition": "12-month price change (risk-adjusted); momentum score",
    "Weighting Method": "Momentum-score weighted, cap per security (e.g. 3%)",
    "Rebalancing": "Semi-annually (e.g. May and November)",
    "Skip Month": "Check prospectus (often 12-month raw or risk-adjusted)",
}

UMD_METHODOLOGY = {
    "Source": "Fama-French, Ken French Data Library",
    "Universe": "All NYSE, AMEX, NASDAQ stocks",
    "Number of Stocks": "~300–400 per leg (600–800 total)",
    "Formation Period": "11 months (t-12 to t-2, skip t-1)",
    "Skip Month": "Yes (skip month t-1)",
    "Ranking": "Deciles by past return",
    "Long Leg": "Top decile (winners)",
    "Short Leg": "Bottom decile (losers)",
    "Weighting": "Equal-weighted within each leg",
    "Construction": "UMD = Winners − Losers",
    "Rebalancing": "Monthly",
    "Market Exposure": "Market-neutral (~0)",
}

# Comparison table (SPMO vs UMD) – used by Q2.2 and report
COMPARISON_FEATURES = [
    "Universe", "Market Cap", "Selection", "Number of Stocks",
    "Long/Short", "Lookback", "Skip Recent Month?", "Weighting",
    "Rebalancing", "Market Exposure",
]
COMPARISON_UMD = [
    "All US stocks (~3000)", "All caps", "Top/Bottom 10% by return",
    "~300–400 per leg", "Long-Short (market neutral)", "11 months (t-12 to t-2)",
    "Yes", "Equal-weighted", "Monthly", "~0",
]
COMPARISON_SPMO = [
    "S&P 500 only (~500)", "Large cap only", "Top by momentum score",
    "~100 stocks", "Long-only", "12 months (verify prospectus)",
    "Verify prospectus", "Momentum-score weighted, cap", "Semi-annually", "~1.0",
]

# SPMO quote for report (from Invesco/S&P). Use ASCII hyphens to avoid PDF encoding issues.
SPMO_QUOTE = (
    "The Invesco S&P 500 Momentum ETF seeks to track the investment results (before fees and expenses) "
    "of the S&P 500 Momentum Index. The Index measures the performance of securities in the S&P 500 "
    "that exhibit the highest momentum characteristics based on price performance and risk. Momentum is "
    "measured using a momentum score (e.g. 12-month price change, risk-adjusted). The Index typically "
    "consists of approximately 100 stocks, reconstituted and rebalanced semi-annually, with weights "
    "by momentum score and a cap per security (e.g. 3%)."
)

# Plot filenames for report PDF (relative to OUT_DIR)
REPORT_PLOT_Q1 = "q2_1_spmo_umd_regression_diagnostics.png"
REPORT_PLOT_Q3 = "q2_3_momentum_decomposition.png"
