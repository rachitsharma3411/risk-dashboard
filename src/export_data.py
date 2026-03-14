"""
export_data.py
--------------
Runs the derivatives risk pipeline and writes the full payload to
../docs/data.json so GitHub Pages can serve a live dashboard.
Called by the GitHub Actions workflow daily after market close.
"""

import json
import os
import sys
from datetime import date, datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))

from data.market_data import get_spot_price, get_risk_free_rate, get_vix, get_market_context
from data.storage import save_snapshot, load_snapshot
from engine.portfolio import load_portfolio, compute_position_metrics, aggregate_portfolio, build_report_payload
import config


def _add_history(docs_path: str, today_payload: dict) -> list:
    history_file = os.path.join(docs_path, "data.json")
    history = []
    if os.path.exists(history_file):
        try:
            with open(history_file) as f:
                existing = json.load(f)
            history = existing.get("history", [])
        except Exception:
            history = []

    pm = today_payload["portfolio_metrics"]
    current = pm.get("current", {})
    history.append({
        "date": today_payload["report_date"],
        "total_MtM": current.get("total_MtM", 0),
        "daily_PnL": current.get("daily_PnL", 0),
        "net_dollar_delta": current.get("net_dollar_delta", 0),
        "net_dollar_vega": current.get("net_dollar_vega", 0),
        "net_dollar_theta": current.get("net_dollar_theta", 0),
    })

    seen = set()
    deduped = []
    for h in reversed(history):
        if h["date"] not in seen:
            seen.add(h["date"])
            deduped.append(h)
    return list(reversed(deduped))[-30:]


def run():
    today = date.today()
    today_str = today.strftime("%Y-%m-%d")
    print(f"[export_data] Running pipeline for {today_str}")

    portfolio_path = os.path.join(os.path.dirname(__file__), config.PORTFOLIO_FILE)
    portfolio_df = load_portfolio(portfolio_path)
    tickers = portfolio_df["ticker"].unique().tolist()

    r = get_risk_free_rate()
    vix = get_vix()
    spot_prices = {t: get_spot_price(t) for t in tickers}
    market_context = get_market_context(tickers)
    market_context["VIX"] = vix

    print(f"[export_data] Spots: {spot_prices}")
    print(f"[export_data] r={r:.4f}, VIX={vix:.2f}")

    positions = []
    for _, row in portfolio_df.iterrows():
        try:
            m = compute_position_metrics(row, spot_prices, r)
            positions.append(m)
        except Exception as e:
            print(f"[export_data] Skipping position: {e}")

    portfolio_metrics = aggregate_portfolio(positions)

    prior_snapshot = None
    db_path = os.path.join(os.path.dirname(__file__), "snapshots.db")
    for offset in range(1, 8):
        candidate = (today - timedelta(days=offset)).strftime("%Y-%m-%d")
        snap = load_snapshot(candidate, db_path=db_path)
        if snap:
            prior_snapshot = snap
            break

    alerts = []
    if abs(portfolio_metrics["net_dollar_delta"]) > config.DELTA_LIMIT:
        alerts.append(f"DELTA LIMIT BREACH: ${portfolio_metrics['net_dollar_delta']:,.0f}")
    if abs(portfolio_metrics["net_dollar_vega"]) > config.VEGA_LIMIT:
        alerts.append(f"VEGA LIMIT BREACH: ${portfolio_metrics['net_dollar_vega']:,.0f}")
    for _, row in portfolio_df.iterrows():
        days_left = (row["expiry"] - today).days
        if 0 <= days_left < 7:
            alerts.append(f"NEAR EXPIRY: {row['ticker']} {row['option_type'].upper()} K={row['strike']:.0f} in {days_left}d")

    payload = build_report_payload(portfolio_metrics, market_context, prior_snapshot, alerts)
    payload["spot_prices"] = spot_prices
    payload["limits"] = {"delta_limit": config.DELTA_LIMIT, "vega_limit": config.VEGA_LIMIT}

    snapshot_data = {
        "portfolio_metrics": payload["portfolio_metrics"],
        "market_context": market_context,
        "alerts": alerts,
        "report_date": today_str,
    }
    save_snapshot(snapshot_data, today_str, db_path=db_path)

    docs_path = os.path.join(os.path.dirname(__file__), "..", "docs")
    os.makedirs(docs_path, exist_ok=True)

    history = _add_history(docs_path, payload)
    payload["history"] = history
    payload["generated_at"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    out_path = os.path.join(docs_path, "data.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"[export_data] Written to {out_path}")
    print(f"[export_data] Total MtM: ${portfolio_metrics['total_MtM']:,.2f}")
    print(f"[export_data] Daily P&L: ${portfolio_metrics['daily_PnL']:,.2f}")


if __name__ == "__main__":
    run()
