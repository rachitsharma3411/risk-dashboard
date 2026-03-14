"""
main.py
-------
Entry point for the Automated Derivatives Risk Report System.

Pipeline
--------
1.  Set up structured logging.
2.  Load configuration and portfolio CSV.
3.  Fetch live market data (spot prices, risk-free rate, VIX, daily moves).
4.  Load prior-day snapshot from SQLite (None on first run).
5.  Compute per-position mark-to-market metrics and Greeks.
6.  Aggregate to portfolio-level risk figures.
7.  Check for limit breaches and near-expiry positions; build alert list.
8.  Assemble the full report payload dict.
9.  Call Claude API to generate AI risk commentary.
10. Create output directory and generate timestamped PDF report.
11. Persist today's snapshot to SQLite for tomorrow's comparison.
12. Print a concise summary to stdout.

Run
---
::

    python main.py

The script exits with code 0 on success and code 1 on unrecoverable errors.
"""

import logging
import os
import sys
import traceback
from datetime import date, datetime

# ---------------------------------------------------------------------------
# 1. Logging setup (before any other imports that use logging)
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("main")

# ---------------------------------------------------------------------------
# Application imports (after logging is configured)
# ---------------------------------------------------------------------------
try:
    import config
    from data.market_data import (
        get_spot_price,
        get_risk_free_rate,
        get_vix,
        get_market_context,
    )
    from data.storage import save_snapshot, load_snapshot
    from engine.portfolio import (
        load_portfolio,
        compute_position_metrics,
        aggregate_portfolio,
        build_report_payload,
    )
    from narrator.claude_narrator import generate_risk_commentary
    from reports.pdf_report import generate_pdf_report
except ImportError as exc:
    logger.critical("Import error — have you run 'pip install -r requirements.txt'? %s", exc)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Alert checking
# ---------------------------------------------------------------------------

def _check_alerts(
    portfolio_metrics: dict,
    portfolio_df,
    today: date,
) -> list[str]:
    """
    Inspect portfolio-level metrics and individual positions for risk alerts.

    Checks performed
    ----------------
    - Net dollar delta exceeds ``config.DELTA_LIMIT``.
    - Net dollar vega exceeds ``config.VEGA_LIMIT``.
    - Any position expires within 7 calendar days.

    Parameters
    ----------
    portfolio_metrics : dict
        Aggregated portfolio metrics from :func:`engine.portfolio.aggregate_portfolio`.
    portfolio_df : pd.DataFrame
        Raw portfolio DataFrame (used to check expiry dates).
    today : date
        Today's date (used for expiry proximity check).

    Returns
    -------
    list[str]
        Human-readable alert strings; empty list if no alerts are triggered.
    """
    alerts: list[str] = []

    net_delta = abs(portfolio_metrics.get("net_dollar_delta", 0.0))
    if net_delta > config.DELTA_LIMIT:
        alerts.append(
            f"DELTA LIMIT BREACH: Net dollar delta ${net_delta:,.0f} "
            f"exceeds limit ${config.DELTA_LIMIT:,.0f}"
        )

    net_vega = abs(portfolio_metrics.get("net_dollar_vega", 0.0))
    if net_vega > config.VEGA_LIMIT:
        alerts.append(
            f"VEGA LIMIT BREACH: Net dollar vega ${net_vega:,.0f} "
            f"exceeds limit ${config.VEGA_LIMIT:,.0f}"
        )

    for _, row in portfolio_df.iterrows():
        expiry = row["expiry"]
        days_left = (expiry - today).days
        if 0 <= days_left < 7:
            alerts.append(
                f"NEAR EXPIRY: {row['ticker']} {row['option_type'].upper()} "
                f"K={row['strike']:.0f} expires in {days_left} day(s) "
                f"on {expiry} (qty={row['quantity']})"
            )

    return alerts


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline() -> int:
    """
    Execute the full derivatives risk report pipeline end-to-end.

    Returns
    -------
    int
        0 on success, 1 on unrecoverable error.
    """
    today = date.today()
    today_str = today.strftime("%Y-%m-%d")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 70)
    logger.info("Derivatives Risk Report Pipeline — %s", today_str)
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # 2. Load configuration and portfolio
    # ------------------------------------------------------------------
    try:
        logger.info("Step 2: Loading portfolio from '%s'...", config.PORTFOLIO_FILE)
        portfolio_df = load_portfolio(config.PORTFOLIO_FILE)
        logger.info("  Loaded %d positions.", len(portfolio_df))
    except Exception:
        logger.critical("Failed to load portfolio:\n%s", traceback.format_exc())
        return 1

    tickers = portfolio_df["ticker"].unique().tolist()

    # ------------------------------------------------------------------
    # 3. Fetch live market data
    # ------------------------------------------------------------------
    try:
        logger.info("Step 3: Fetching live market data...")

        logger.info("  Fetching risk-free rate (%s)...", config.RISK_FREE_RATE_TICKER)
        r = get_risk_free_rate()
        logger.info("  Risk-free rate: %.4f (%.2f%%)", r, r * 100)

        logger.info("  Fetching VIX...")
        vix = get_vix()
        logger.info("  VIX: %.2f", vix)

        logger.info("  Fetching spot prices for: %s", tickers)
        spot_prices: dict = {}
        for t in tickers:
            try:
                spot_prices[t] = get_spot_price(t)
                logger.info("    %s: $%.2f", t, spot_prices[t])
            except Exception as exc:
                logger.warning("    Could not fetch spot for %s: %s", t, exc)
                spot_prices[t] = 0.0

        logger.info("  Fetching daily market context...")
        market_context = get_market_context(tickers)
        market_context["VIX"] = vix  # embed VIX in context dict
        for t, pct in market_context.items():
            if t == "VIX":
                logger.info("    VIX level: %.2f", pct)
            else:
                logger.info("    %s daily move: %+.2f%%", t, pct * 100)

    except Exception:
        logger.critical("Failed during market data fetch:\n%s", traceback.format_exc())
        return 1

    # ------------------------------------------------------------------
    # 4. Load prior snapshot
    # ------------------------------------------------------------------
    try:
        logger.info("Step 4: Loading prior-day snapshot...")
        prior_date = None
        # Try to find the most recent snapshot by checking yesterday first
        from datetime import timedelta
        for offset in range(1, 8):
            candidate = (today - timedelta(days=offset)).strftime("%Y-%m-%d")
            snap = load_snapshot(candidate)
            if snap is not None:
                prior_date = candidate
                prior_snapshot = snap
                break
        else:
            prior_snapshot = None

        if prior_snapshot is None:
            logger.info("  No prior snapshot found (first run — P&L comparison will be zero).")
        else:
            logger.info("  Loaded prior snapshot from %s.", prior_date)
    except Exception:
        logger.warning("Could not load prior snapshot:\n%s", traceback.format_exc())
        prior_snapshot = None

    # ------------------------------------------------------------------
    # 5. Compute per-position metrics
    # ------------------------------------------------------------------
    try:
        logger.info("Step 5: Computing per-position metrics...")
        positions: list[dict] = []
        for idx, row in portfolio_df.iterrows():
            try:
                metrics = compute_position_metrics(row, spot_prices, r)
                positions.append(metrics)
                logger.info(
                    "  [%d] %s %s K=%.0f exp=%s  MtM=$%.0f  σ=%.1f%%  Δ$=%.0f  V$=%.0f",
                    idx,
                    metrics["ticker"],
                    metrics["option_type"].upper(),
                    metrics["strike"],
                    metrics["expiry"],
                    metrics["MtM"],
                    metrics["sigma"] * 100,
                    metrics["dollar_delta"],
                    metrics["dollar_vega"],
                )
            except Exception as exc:
                logger.warning("  Skipping position %d due to error: %s", idx, exc)
    except Exception:
        logger.critical("Failed during position metrics computation:\n%s", traceback.format_exc())
        return 1

    # ------------------------------------------------------------------
    # 6. Aggregate portfolio
    # ------------------------------------------------------------------
    try:
        logger.info("Step 6: Aggregating portfolio-level risk...")
        portfolio_metrics = aggregate_portfolio(positions)
        logger.info("  Total MtM:        $%s", f"{portfolio_metrics['total_MtM']:,.0f}")
        logger.info("  Daily P&L:        $%s", f"{portfolio_metrics['daily_PnL']:,.0f}")
        logger.info("  Net $ Delta:      $%s", f"{portfolio_metrics['net_dollar_delta']:,.0f}")
        logger.info("  Net $ Vega:       $%s", f"{portfolio_metrics['net_dollar_vega']:,.0f}")
        logger.info("  Net $ Theta:      $%s", f"{portfolio_metrics['net_dollar_theta']:,.0f}")
    except Exception:
        logger.critical("Portfolio aggregation failed:\n%s", traceback.format_exc())
        return 1

    # ------------------------------------------------------------------
    # 7. Check alerts
    # ------------------------------------------------------------------
    try:
        logger.info("Step 7: Checking risk alerts...")
        alerts = _check_alerts(portfolio_metrics, portfolio_df, today)
        if alerts:
            logger.warning("  %d alert(s) triggered:", len(alerts))
            for a in alerts:
                logger.warning("    ! %s", a)
        else:
            logger.info("  No alerts triggered.")
    except Exception:
        logger.warning("Alert checking failed:\n%s", traceback.format_exc())
        alerts = []

    # ------------------------------------------------------------------
    # 8. Build report payload
    # ------------------------------------------------------------------
    try:
        logger.info("Step 8: Building report payload...")
        payload = build_report_payload(portfolio_metrics, market_context, prior_snapshot, alerts)
        logger.info("  Payload assembled (%d positions).", len(positions))
    except Exception:
        logger.critical("Failed to build report payload:\n%s", traceback.format_exc())
        return 1

    # ------------------------------------------------------------------
    # 9. Generate AI risk commentary
    # ------------------------------------------------------------------
    try:
        logger.info("Step 9: Generating Claude risk commentary...")
        if config.CLAUDE_API_KEY:
            commentary = generate_risk_commentary(payload)
            logger.info("  Commentary generated (%d characters).", len(commentary))
        else:
            logger.warning("  ANTHROPIC_API_KEY not set — skipping AI commentary.")
            commentary = (
                "AI commentary is disabled because the ANTHROPIC_API_KEY "
                "environment variable is not set. Set it and re-run the pipeline "
                "to include Claude-generated risk narrative."
            )
    except Exception:
        logger.warning("Commentary generation failed:\n%s", traceback.format_exc())
        commentary = "[Commentary generation failed — see logs for details.]"

    # ------------------------------------------------------------------
    # 10. Generate PDF report
    # ------------------------------------------------------------------
    try:
        logger.info("Step 10: Generating PDF report...")
        os.makedirs(config.REPORT_OUTPUT_DIR, exist_ok=True)
        filename = f"derivatives_risk_report_{timestamp}.pdf"
        output_path = os.path.join(config.REPORT_OUTPUT_DIR, filename)
        generate_pdf_report(payload, commentary, output_path)
        logger.info("  PDF saved: %s", output_path)
    except Exception:
        logger.error("PDF generation failed:\n%s", traceback.format_exc())
        output_path = None  # non-fatal: continue to save snapshot

    # ------------------------------------------------------------------
    # 11. Save snapshot
    # ------------------------------------------------------------------
    try:
        logger.info("Step 11: Saving daily snapshot to SQLite...")
        snapshot_data = {
            "portfolio_metrics": payload["portfolio_metrics"],
            "market_context": market_context,
            "alerts": alerts,
            "report_date": today_str,
            "vix": vix,
        }
        save_snapshot(snapshot_data, today_str)
        logger.info("  Snapshot saved for %s.", today_str)
    except Exception:
        logger.warning("Snapshot save failed:\n%s", traceback.format_exc())

    # ------------------------------------------------------------------
    # 12. Print summary
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("  DERIVATIVES RISK REPORT — SUMMARY")
    print("=" * 60)
    print(f"  Report Date   : {today_str}")
    print(f"  Positions     : {len(positions)}")
    print(f"  Total MtM     : ${portfolio_metrics['total_MtM']:>12,.2f}")
    print(f"  Daily P&L     : ${portfolio_metrics['daily_PnL']:>12,.2f}")
    print(f"  Net $ Delta   : ${portfolio_metrics['net_dollar_delta']:>12,.2f}")
    print(f"  Net $ Vega    : ${portfolio_metrics['net_dollar_vega']:>12,.2f}")
    print(f"  Net $ Gamma   : ${portfolio_metrics['net_dollar_gamma']:>12,.4f}")
    print(f"  Net $ Theta   : ${portfolio_metrics['net_dollar_theta']:>12,.2f}")
    print(f"  VIX           : {vix:>12.2f}")
    print(f"  Alerts        : {len(alerts)}")
    for a in alerts:
        print(f"    ! {a}")
    if output_path:
        print(f"  PDF Report    : {output_path}")
    else:
        print("  PDF Report    : [FAILED — see logs]")
    print("=" * 60)
    print()

    logger.info("Pipeline complete.")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(run_pipeline())
