import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd

from q2_config import (
    COMPARISON_FEATURES,
    COMPARISON_SPMO,
    COMPARISON_UMD,
    OUT_DIR,
    OTHER_ETF_TICKERS,
    REPORT_PLOT_Q1,
    REPORT_PLOT_Q3,
    SPMO_QUOTE,
)


def _get_q1():
    p = os.path.join(OUT_DIR, "q2_1_regression_summary.csv")
    if not os.path.isfile(p):
        return None
    df = pd.read_csv(p)
    out = {}
    for _, row in df.iterrows():
        m, v = row["Metric"], row["Value"]
        if m == "Beta (UMD)":
            out["beta_umd"] = float(v)
        elif m == "Alpha (annualized)":
            out["alpha_annual"] = float(v)
        elif m == "R-squared":
            out["r_squared"] = float(v)
        elif m == "N":
            out["observations"] = int(float(v))
        elif m == "Start":
            out["start"] = str(v)
        elif m == "End":
            out["end"] = str(v)
    return out if out else None


def _get_q3():
    p = os.path.join(OUT_DIR, "q2_3_all_models_summary.csv")
    if not os.path.isfile(p):
        return None
    return pd.read_csv(p)


def _get_q4():
    p = os.path.join(OUT_DIR, "q2_4_ff6_regression_results.csv")
    if not os.path.isfile(p):
        return None
    return pd.read_csv(p)


def _get_q5():
    p = os.path.join(OUT_DIR, "q2_5_other_etfs_ff6.csv")
    if not os.path.isfile(p):
        return []
    df = pd.read_csv(p)
    return df.to_dict("records")


def build_md():
    """Build report markdown from files and config. Returns string."""
    q1 = _get_q1()
    q3 = _get_q3()
    q4 = _get_q4()
    q5 = _get_q5()
    lines = []
    lines.append("# MFIN 7037 Homework 1 – Question 2: Smart Beta ETFs (SPMO)")
    lines.append("")
    # --- 1 ---
    lines.append("## 1) What is the beta of SPMO to the UMD factor? Does this mean the ETF is broken?")
    lines.append("")
    if q1:
        n = q1.get("observations", 0)
        start = q1.get("start", "")
        end = q1.get("end", "")
        lines.append(f"The beta of SPMO to the UMD factor (from a regression of SPMO excess return on Mkt-RF and UMD, controlling for market) is **{q1['beta_umd']:.4f}** over {start} to {end} (N={n} months); R² is **{q1['r_squared']:.4f}** and alpha (annualized) is **{q1['alpha_annual']:.2%}**. This shows meaningful momentum exposure. A bivariate regression of SPMO on UMD alone gives a misleadingly low beta because SPMO is mostly market and UMD is market-neutral, so the market-controlled regression is the appropriate specification.")
        lines.append("")
        lines.append(f"**Does this mean the ETF is broken?** **No.** A beta to UMD below 1 (here {q1['beta_umd']:.2f}) is expected because: UMD is long-short while SPMO is long-only (one leg), SPMO has full market exposure (S&P 500) while UMD is market-neutral, and universe, weighting, and rebalancing differ from academic UMD.")
    else:
        lines.append("(Run `q2_1_spmo_umd_beta.py` to populate.)")
    lines.append("")
    lines.append("---")
    lines.append("")
    # --- 2 ---
    lines.append("## 2) Read the SPMO definition, extract a quote. How does this differ from the original UMD construction?")
    lines.append("")
    lines.append("**Quote (from Invesco / S&P index methodology):**")
    lines.append("")
    lines.append("> " + SPMO_QUOTE)
    lines.append("")
    lines.append("**How this differs from UMD:**")
    lines.append("")
    lines.append("| Dimension | UMD (Fama–French) | SPMO |")
    lines.append("|-----------|-------------------|------|")
    for i in range(len(COMPARISON_FEATURES)):
        lines.append(f"| {COMPARISON_FEATURES[i]} | {COMPARISON_UMD[i]} | {COMPARISON_SPMO[i]} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    # --- 3 ---
    lines.append("## 3) Extra credit: Beta to long-leg; construct long/short; VW vs EW; consistency")
    lines.append("")
    if q3 is not None and len(q3):
        def val(df, model, col):
            r = df[df["Model"] == model]
            return r[col].iloc[0] if len(r) else None
        beta_vw = val(q3, "Winners_VW", "Beta")
        beta_ew = val(q3, "Winners_EW", "Beta")
        r2_vw = val(q3, "Winners_VW", "R-squared")
        r2_ew = val(q3, "Winners_EW", "R-squared")
        if beta_vw is not None:
            lines.append(f"- **Beta to long leg:** Winners_VW β ≈ {beta_vw:.3f} (R² ≈ {r2_vw:.3f}); Winners_EW β ≈ {beta_ew:.3f} (R² ≈ {r2_ew:.3f}).")
        lines.append("")
        lines.append("The beta to the long leg (winners) is the sensitivity of SPMO to the top-momentum decile. A beta below 1 means SPMO behaves like a diluted or capped version of the winners portfolio (e.g. ~100 names, cap per name). The higher R² for Winners_VW than Winners_EW means SPMO’s returns are better explained by value-weighted winners—consistent with an S&P 500, momentum-score-weighted product. Regressing SPMO on MomLS (winners minus losers) gives small or negative beta because SPMO is long-only: it holds winners but not the short leg, so it does not move with the pure long–short factor.")
        lines.append("- **Constructed long/short:** MomLS = Winners (D10) − Losers (D1). SPMO on MomLS gives small/negative betas (long-only).")
        lines.append("- **VW vs EW:** SPMO tracks **value-weighted** momentum more closely; **consistent** with S&P 500 and momentum-score weighting.")
    else:
        lines.append("(Run `q2_3_long_leg.py` to populate.)")
    lines.append("")
    lines.append("---")
    lines.append("")
    # --- 4 ---
    lines.append("## 4) Control for Fama–French factors. Map to SPMO definition. Does correcting for long-bias fix market beta? Size-bias?")
    lines.append("")
    if q4 is not None and len(q4):
        def fval(df, factor, col):
            r = df[df["Factor"] == factor]
            return r[col].iloc[0] if len(r) else None
        mkt = fval(q4, "Mkt-RF", "Beta")
        smb = fval(q4, "SMB", "Beta")
        umd = fval(q4, "UMD", "Beta")
        alpha = fval(q4, "Alpha", "Beta")
        if mkt is not None:
            alpha_ann = ((1 + alpha) ** 12 - 1) * 100 if alpha is not None else 0
            lines.append(f"- **FF6:** Market beta ≈ {mkt:.3f}, SMB ≈ {smb:.3f}, UMD ≈ {umd:.3f}; alpha (annual) ≈ {alpha_ann:.2f}%.")
        lines.append("")
        lines.append("Market beta (Mkt-RF) near 1 means SPMO has roughly full exposure to the equity market—expected for a long-only S&P 500 subset. SMB (small minus big) negative means a large-cap tilt, which matches the S&P 500 universe. UMD positive means the strategy still has a positive loading on the academic momentum factor after controlling for market and size. Alpha is the average monthly return not explained by the six factors; annualized alpha around 1–2% is modest and can reflect fees, implementation, or other tilts. Controlling for factors does not remove market beta: it only isolates it. So the long-bias (market exposure) is still there; the regression just separates it from momentum and size.")
        lines.append("- **Mapping:** Market beta near 1 = S&P 500 exposure; negative SMB = large-cap only. Size bias **makes sense**.")
        lines.append("- **Long-bias:** Controlling for factors does **not** remove market beta; it stays ~1. Correcting isolates exposure; it doesn’t remove it.")
    else:
        lines.append("(Run `q2_4_ff6_controls.py` to populate.)")
    lines.append("")
    lines.append("---")
    lines.append("")
    # --- 5 ---
    lines.append("## 5) Two other momentum ETFs. FF6 loadings vs index construction? Opaque?")
    lines.append("")
    ticker_to_name = dict(OTHER_ETF_TICKERS)
    if q5:
        for r in q5:
            ticker = r.get("ticker", "")
            name = ticker_to_name.get(ticker, ticker)
            if "error" in r:
                lines.append(f"- **{ticker}** ({name}): data error.")
            else:
                lines.append(f"- **{ticker}** ({name}): Mkt-RF ≈ {r.get('Mkt-RF', 0):.3f}, SMB ≈ {r.get('SMB', 0):.3f}, HML ≈ {r.get('HML', 0):.3f}, UMD ≈ {r.get('UMD', 0):.3f}, R² ≈ {r.get('R2', 0):.3f}.")
                lines.append("  Mkt-RF is market exposure (≈1 for long-only). SMB > 0 = small-cap tilt, SMB < 0 = large-cap. HML is value tilt. UMD is exposure to the momentum factor; positive UMD is expected for momentum ETFs. R² shows how much of the ETF’s return variance is explained by the six factors; high R² means the strategy is well described by these exposures.")
                if ticker == "MTUM":
                    lines.append("  - **Reconciliation:** Large-cap momentum; market beta ~1 and negative SMB consistent; construction transparent.")
                elif ticker == "QMOM":
                    lines.append("  - **Reconciliation:** Quantitative momentum; positive SMB = small-cap tilt; loadings partly opaque but consistent with momentum + market.")
                else:
                    lines.append("  - **Reconciliation:** Long-only momentum; market ~1, positive UMD.")
    else:
        lines.append("(Run `q2_5_other_etfs.py` to populate.)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Report built from q2_* CSV outputs. Generated by q2_report.py.*")
    return "\n".join(lines)


def _b(s):
    """Wrap text for reportlab bold (escape & <> for XML)."""
    s = str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"<b>{s}</b>"


def build_pdf(pdf_path):
    """Build PDF programmatically: proper tables, bold text, embedded plots."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError:
        print("Install reportlab for PDF: pip install reportlab")
        return False

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    styles = getSampleStyleSheet()
    style_title = ParagraphStyle(
        name="ReportTitle",
        parent=styles["Heading1"],
        fontSize=18,
        spaceAfter=14,
        alignment=1,
    )
    style_h2 = ParagraphStyle(
        name="H2",
        parent=styles["Heading2"],
        fontSize=12,
        spaceAfter=6,
        spaceBefore=12,
    )
    body = styles["Normal"]
    body.spaceAfter = 4
    table_text = ParagraphStyle(name="TableText", parent=body, fontSize=9, spaceAfter=0)
    story = []

    q1 = _get_q1()
    q3 = _get_q3()
    q4 = _get_q4()
    q5 = _get_q5()

    # Title
    story.append(Paragraph("MFIN 7037 Homework 1 – Question 2: Smart Beta ETFs (SPMO)", style_title))
    story.append(Spacer(1, 6))

    # ----- Section 1 -----
    story.append(Paragraph(
        "1) What is the beta of SPMO to the UMD factor? Does this mean the ETF is broken?",
        style_h2,
    ))
    if q1:
        n = q1.get("observations", 0)
        start = q1.get("start", "")
        end = q1.get("end", "")
        story.append(Paragraph(
            f"• {_b('Beta of SPMO to UMD:')} {q1['beta_umd']:.4f} (from regression of SPMO excess on Mkt-RF and UMD; see q2_1_regression_summary.csv).",
            body,
        ))
        story.append(Paragraph(
            f"• {_b('R²:')} {q1['r_squared']:.4f}; {_b('Alpha (annualized):')} {q1['alpha_annual']:.2%}. "
            f"Sample: {start} to {end} (N={n} months).",
            body,
        ))
        story.append(Paragraph(
            "This beta is from a market-controlled regression and shows meaningful momentum exposure. A bivariate regression of SPMO on UMD alone gives a misleadingly low beta because SPMO is mostly market and UMD is market-neutral; the market-controlled specification is the appropriate one.",
            body,
        ))
    else:
        story.append(Paragraph("(Run q2_1_spmo_umd_beta.py to populate.)", body))
    story.append(Paragraph(
        f"{_b('Does this mean the ETF is broken?')} {_b('No.')} A beta to UMD below 1 is expected because:",
        body,
    ))
    story.append(Paragraph("• UMD is long–short; SPMO is long-only, so it captures roughly one leg.", body))
    story.append(Paragraph("• SPMO has market exposure (S&P 500); UMD is market-neutral.", body))
    story.append(Paragraph("• Universe, weighting, and rebalancing differ from academic UMD.", body))
    # Plot Q2.1
    _add_plot(story, body, OUT_DIR, REPORT_PLOT_Q1, "Figure 1: SPMO vs UMD regression diagnostics.")
    story.append(Spacer(1, 8))

    # ----- Section 2 -----
    story.append(Paragraph(
        "2) Read the SPMO definition, extract a quote. How does this differ from the original UMD construction?",
        style_h2,
    ))
    story.append(Paragraph(_b("Quote (from Invesco / S&P index methodology):"), body))
    story.append(Paragraph('"' + SPMO_QUOTE + '"', body))
    story.append(Paragraph(_b("How this differs from UMD:"), body))
    # Comparison table (Paragraph in cells so header bold renders)
    table_data = [
        [Paragraph(_b("Dimension"), table_text), Paragraph(_b("UMD (Fama–French)"), table_text), Paragraph(_b("SPMO"), table_text)],
    ]
    for i in range(len(COMPARISON_FEATURES)):
        table_data.append([COMPARISON_FEATURES[i], COMPARISON_UMD[i], COMPARISON_SPMO[i]])
    t = Table(table_data, colWidths=[1.4 * inch, 2.4 * inch, 2.4 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("TOPPADDING", (0, 0), (-1, 0), 8),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(t)
    story.append(Spacer(1, 8))

    # ----- Section 3 -----
    story.append(Paragraph(
        "3) Extra credit: Beta to long-leg; construct long/short; VW vs EW; consistency",
        style_h2,
    ))
    if q3 is not None and len(q3):
        def val(df, model, col):
            r = df[df["Model"] == model]
            return r[col].iloc[0] if len(r) else None
        beta_vw = val(q3, "Winners_VW", "Beta")
        beta_ew = val(q3, "Winners_EW", "Beta")
        r2_vw = val(q3, "Winners_VW", "R-squared")
        r2_ew = val(q3, "Winners_EW", "R-squared")
        if beta_vw is not None:
            story.append(Paragraph(
                f"• {_b('Beta to long leg:')} Winners_VW β ≈ {beta_vw:.3f} (R² ≈ {r2_vw:.3f}); "
                f"Winners_EW β ≈ {beta_ew:.3f} (R² ≈ {r2_ew:.3f}).",
                body,
            ))
        story.append(Paragraph(
            "The beta to the long leg (winners) is the sensitivity of SPMO to the top-momentum "
            "decile. A beta below 1 means SPMO behaves like a diluted or capped version of the winners portfolio (e.g. ~100 names, "
            "cap per name). The higher R² for Winners_VW than Winners_EW means SPMO's returns are better explained by value-weighted "
            "winners—consistent with an S&P 500, momentum-score-weighted product. Regressing SPMO on MomLS (winners minus losers) "
            "gives small or negative beta because SPMO is long-only: it holds winners but not the short leg, so it does not move "
            "with the pure long–short factor.",
            body,
        ))
    story.append(Paragraph(
        "• Constructed long/short: MomLS = Winners (D10) − Losers (D1). "
        "SPMO on MomLS gives small/negative betas (long-only).",
        body,
    ))
    story.append(Paragraph(
        "• VW vs EW: SPMO tracks value-weighted momentum more closely; consistent with S&P 500 and momentum-score weighting.",
        body,
    ))
    _add_plot(story, body, OUT_DIR, REPORT_PLOT_Q3, "Figure 2: Momentum decomposition (betas and R²).")
    story.append(Spacer(1, 8))

    # ----- Section 4 -----
    story.append(Paragraph(
        "4) Control for Fama–French factors. Map to SPMO definition. Does correcting for long-bias fix market beta? Size-bias?",
        style_h2,
    ))
    if q4 is not None and len(q4):
        def fval(df, factor, col):
            r = df[df["Factor"] == factor]
            return r[col].iloc[0] if len(r) else None
        mkt = fval(q4, "Mkt-RF", "Beta")
        smb = fval(q4, "SMB", "Beta")
        umd = fval(q4, "UMD", "Beta")
        alpha = fval(q4, "Alpha", "Beta")
        if mkt is not None:
            alpha_ann = ((1 + alpha) ** 12 - 1) * 100 if alpha is not None else 0
            story.append(Paragraph(
                f"• {_b('FF6:')} Market beta ≈ {mkt:.3f}, SMB ≈ {smb:.3f}, UMD ≈ {umd:.3f}; alpha (annual) ≈ {alpha_ann:.2f}%.",
                body,
            ))
            story.append(Paragraph(
                "Market beta (Mkt-RF) near 1 means SPMO has roughly full exposure to the equity "
                "market—expected for a long-only S&P 500 subset. SMB (small minus big) negative means a large-cap tilt, which matches "
                "the S&P 500 universe. UMD positive means the strategy still has a positive loading on the academic momentum factor "
                "after controlling for market and size. Alpha is the average monthly return not explained by the six factors; annualized "
                "alpha around 1–2% is modest and can reflect fees, implementation, or other tilts. Controlling for factors does not "
                "remove market beta: it only isolates it. So the long-bias (market exposure) is still there; the regression just "
                "separates it from momentum and size.",
                body,
            ))
        story.append(Paragraph(
            f"• {_b('Mapping:')} Market beta near 1 = S&P 500 exposure; negative SMB = large-cap only. Size bias makes sense.",
            body,
        ))
        story.append(Paragraph(
            "• Long-bias: Controlling for factors does not remove market beta; it stays ~1. Correcting isolates exposure; it doesn't remove it.",
            body,
        ))
    else:
        story.append(Paragraph("(Run q2_4_ff6_controls.py to populate.)", body))
    story.append(Spacer(1, 8))

    # ----- Section 5 -----
    story.append(Paragraph(
        "5) Two other momentum ETFs. FF6 loadings vs index construction? Opaque?",
        style_h2,
    ))
    ticker_to_name = dict(OTHER_ETF_TICKERS)
    if q5:
        for r in q5:
            ticker = r.get("ticker", "")
            name = ticker_to_name.get(ticker, ticker)
            if "error" in r:
                story.append(Paragraph(f"• {_b(ticker)} ({name}): data error.", body))
            else:
                story.append(Paragraph(
                    f"• {_b(ticker)} ({name}): Mkt-RF ≈ {r.get('Mkt-RF', 0):.3f}, SMB ≈ {r.get('SMB', 0):.3f}, "
                    f"HML ≈ {r.get('HML', 0):.3f}, UMD ≈ {r.get('UMD', 0):.3f}, R² ≈ {r.get('R2', 0):.3f}.",
                    body,
                ))
                story.append(Paragraph(
                    f"Mkt-RF is market exposure (≈1 for long-only). SMB &gt; 0 = small-cap tilt, "
                    "SMB &lt; 0 = large-cap. HML is value tilt. UMD is exposure to the momentum factor; positive UMD is expected for "
                    "momentum ETFs. R² shows how much of the ETF's return variance is explained by the six factors; high R² means the "
                    "strategy is well described by these exposures.",
                    body,
                ))
                if ticker == "MTUM":
                    story.append(Paragraph(
                        "  Reconciliation: Large-cap momentum; market beta ~1 and negative SMB consistent; construction transparent.",
                        body,
                    ))
                elif ticker == "QMOM":
                    story.append(Paragraph(
                        "  Reconciliation: Quantitative momentum; positive SMB = small-cap tilt; loadings partly opaque but consistent with momentum + market.",
                        body,
                    ))
                else:
                    story.append(Paragraph("  Reconciliation: Long-only momentum; market ~1, positive UMD.", body))
    else:
        story.append(Paragraph("(Run q2_5_other_etfs.py to populate.)", body))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "Report built from q2_* CSV outputs. Generated by q2_report.py.",
        ParagraphStyle(name="Footer", parent=body, fontSize=8, textColor=colors.grey),
    ))
    doc.build(story)
    return True


def _add_plot(story, body_style, out_dir, filename, caption):
    """Append image and optional caption to story if file exists."""
    path = os.path.join(out_dir, filename)
    if not os.path.isfile(path):
        return
    try:
        from reportlab.lib.units import inch
        from reportlab.platypus import Image as RLImage, Paragraph, Spacer
        img = RLImage(path)
        max_w, max_h = 6 * inch, 4.5 * inch
        if img.drawWidth > max_w or img.drawHeight > max_h:
            ratio = min(max_w / img.drawWidth, max_h / img.drawHeight)
            img.drawWidth *= ratio
            img.drawHeight *= ratio
        story.append(Spacer(1, 6))
        story.append(img)
        if caption:
            from reportlab.lib.styles import ParagraphStyle
            cap_style = ParagraphStyle(name="Caption", parent=body_style, fontSize=9, textColor=(0.3, 0.3, 0.3))
            story.append(Paragraph(caption, cap_style))
        story.append(Spacer(1, 6))
    except Exception:
        pass


def main():
    print("=" * 60)
    print("Q2 Report: building REPORT_Q2.md and REPORT_Q2.pdf")
    print("=" * 60)
    md_content = build_md()
    md_path = os.path.join(OUT_DIR, "REPORT_Q2.md")
    pdf_path = os.path.join(OUT_DIR, "REPORT_Q2.pdf")
    with open(md_path, "w") as f:
        f.write(md_content)
    print("Wrote:", md_path)
    if build_pdf(pdf_path):
        print("Wrote:", pdf_path)
    else:
        print("PDF not built (install reportlab). You can print MD to PDF from your editor.")
    print("\nDone. Outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
