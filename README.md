# Derivatives Risk Dashboard

Automated derivatives risk report system with a live GitHub Pages dashboard.

**Live dashboard:** https://rachitsharma3411.github.io/risk-dashboard/

## What it does

Full automated pipeline:
1. Fetches live market data (yfinance — spot prices, VIX, risk-free rate)
2. Prices all options with Black-Scholes and solves implied vols
3. Computes Greeks (delta, gamma, vega, theta, rho) and P&L attribution
4. Checks risk limits and near-expiry alerts
5. Calls Claude API for institutional-quality risk commentary
6. Generates a timestamped PDF report
7. Writes `docs/data.json` so the GitHub Pages dashboard auto-updates

Runs automatically Mon–Fri at 4:30 PM ET via GitHub Actions.

## Setup

```bash
cd src
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
python main.py
```

PDF reports are saved to `src/reports/output/`.

## GitHub Actions (auto-update)

The workflow `.github/workflows/daily_report.yml` runs `export_data.py` daily after market close and commits the updated `docs/data.json` — the dashboard refreshes automatically.

Add your Anthropic key as a repository secret:
- Settings → Secrets and variables → Actions → `ANTHROPIC_API_KEY`

## Portfolio format (`src/portfolio.csv`)

| Column | Description |
|--------|-------------|
| ticker | Underlying symbol (e.g. SPY, AAPL) |
| option_type | `call` or `put` |
| strike | Strike price |
| expiry | Expiration date YYYY-MM-DD |
| quantity | Contracts (negative = short) |
| entry_price | Premium paid/received at inception |

## Output

- **PDF report** — `src/reports/output/derivatives_risk_report_YYYYMMDD_HHMMSS.pdf`
- **Live dashboard** — https://rachitsharma3411.github.io/risk-dashboard/
- **Snapshot DB** — `src/snapshots.db` (SQLite, 30-day history)

## Risk limits

| Metric | Limit |
|--------|-------|
| Net dollar delta | $100,000 |
| Net dollar vega | $200,000 |
