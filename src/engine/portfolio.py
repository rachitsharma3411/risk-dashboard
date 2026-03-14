"""
engine/portfolio.py
-------------------
Portfolio loading, per-position metrics computation, and aggregate risk
roll-up for the daily derivatives risk report.
"""

import logging
from datetime import date, datetime
from typing import Optional

import pandas as pd

from config import OPTION_MULTIPLIER
from engine.pricing import black_scholes, implied_vol
from engine.greeks import delta, gamma, vega, theta

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Portfolio loading
# ---------------------------------------------------------------------------

def load_portfolio(filepath: str) -> pd.DataFrame:
    """
    Read the portfolio CSV file and return a cleaned DataFrame.

    Expected columns
    ----------------
    ticker : str
        Underlying symbol (e.g. "SPY").
    option_type : str
        "call" or "put".
    strike : float
        Option strike price.
    expiry : str
        Expiration date as "YYYY-MM-DD".
    quantity : int
        Number of contracts (negative = short).
    entry_price : float
        Option premium paid / received at trade inception.

    Parameters
    ----------
    filepath : str
        Path to the portfolio CSV file.

    Returns
    -------
    pd.DataFrame
        Validated portfolio with correct dtypes.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If required columns are missing.
    """
    df = pd.read_csv(filepath)

    required_cols = {"ticker", "option_type", "strike", "expiry", "quantity", "entry_price"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Portfolio CSV missing columns: {missing}")

    df["strike"] = df["strike"].astype(float)
    df["quantity"] = df["quantity"].astype(int)
    df["entry_price"] = df["entry_price"].astype(float)
    df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
    df["option_type"] = df["option_type"].str.lower().str.strip()

    logger.info("Portfolio loaded: %d positions from %s", len(df), filepath)
    return df


# ---------------------------------------------------------------------------
# Per-position metrics
# ---------------------------------------------------------------------------

def compute_position_metrics(
    row: pd.Series,
    spot_prices: dict,
    r: float,
) -> dict:
    """
    Compute full mark-to-market metrics for a single portfolio position.

    The function derives the implied vol from the *entry_price* and then
    prices the option at current market conditions to produce MtM and P&L.

    Parameters
    ----------
    row : pd.Series
        A single row from the portfolio DataFrame (must contain ticker,
        option_type, strike, expiry, quantity, entry_price).
    spot_prices : dict
        Mapping ``{ticker: spot_price}`` for all portfolio underlyings.
    r : float
        Current risk-free rate (decimal, annualised).

    Returns
    -------
    dict
        Keys: ticker, option_type, strike, expiry, quantity, entry_price,
        S, T, sigma, current_price, MtM, unrealized_PnL,
        dollar_delta, dollar_gamma, dollar_vega, dollar_theta.

    Notes
    -----
    - ``T = max((expiry - today).days / 365, 1/365)`` so expiring options
      always have at least one day of T.
    - Dollar Greeks = Greek * quantity * OPTION_MULTIPLIER.
    """
    ticker = row["ticker"]
    opt_type = row["option_type"]
    K = float(row["strike"])
    expiry = row["expiry"]
    qty = int(row["quantity"])
    entry_price = float(row["entry_price"])

    # --- Spot and time ---
    S = float(spot_prices.get(ticker, 0.0))
    if S <= 0:
        logger.warning("No spot price for %s; skipping Greeks", ticker)
        S = K  # fallback to ATM

    today = date.today()
    days_to_expiry = (expiry - today).days
    T = max(days_to_expiry / 365.0, 1.0 / 365.0)

    # --- Implied vol from entry price ---
    sigma = implied_vol(entry_price, S, K, T, r, opt_type)

    # --- Current model price ---
    current_price = black_scholes(S, K, T, r, sigma, opt_type)

    # --- P&L ---
    MtM = current_price * qty * OPTION_MULTIPLIER
    unrealized_PnL = (current_price - entry_price) * qty * OPTION_MULTIPLIER

    # --- Greeks ---
    d = delta(S, K, T, r, sigma, opt_type)
    g = gamma(S, K, T, r, sigma)
    v = vega(S, K, T, r, sigma)
    th = theta(S, K, T, r, sigma, opt_type)

    dollar_delta = d * qty * OPTION_MULTIPLIER
    dollar_gamma = g * qty * OPTION_MULTIPLIER
    dollar_vega = v * qty * OPTION_MULTIPLIER
    dollar_theta = th * qty * OPTION_MULTIPLIER

    return {
        "ticker": ticker,
        "option_type": opt_type,
        "strike": K,
        "expiry": str(expiry),
        "quantity": qty,
        "entry_price": entry_price,
        "S": S,
        "T": T,
        "sigma": sigma,
        "current_price": current_price,
        "MtM": MtM,
        "unrealized_PnL": unrealized_PnL,
        "dollar_delta": dollar_delta,
        "dollar_gamma": dollar_gamma,
        "dollar_vega": dollar_vega,
        "dollar_theta": dollar_theta,
    }


# ---------------------------------------------------------------------------
# Aggregate portfolio
# ---------------------------------------------------------------------------

def aggregate_portfolio(positions: list[dict]) -> dict:
    """
    Roll up per-position metrics into portfolio-level risk figures.

    P&L Attribution
    ---------------
    - ``delta_pnl``  ≈ net_dollar_delta × average_spot_pct_move (proxy: 0)
    - ``vega_pnl``   ≈ net_dollar_vega (best estimate without intraday vega)
    - ``theta_pnl``  = sum of dollar_theta across all positions

    Parameters
    ----------
    positions : list[dict]
        List of dicts as returned by :func:`compute_position_metrics`.

    Returns
    -------
    dict
        Keys: net_dollar_delta, net_dollar_gamma, net_dollar_vega,
        net_dollar_theta, total_MtM, daily_PnL, PnL_attribution,
        positions (the raw list), position_count.
    """
    if not positions:
        return {
            "net_dollar_delta": 0.0,
            "net_dollar_gamma": 0.0,
            "net_dollar_vega": 0.0,
            "net_dollar_theta": 0.0,
            "total_MtM": 0.0,
            "daily_PnL": 0.0,
            "PnL_attribution": {"delta_pnl": 0.0, "vega_pnl": 0.0, "theta_pnl": 0.0},
            "positions": [],
            "position_count": 0,
        }

    net_dollar_delta = sum(p["dollar_delta"] for p in positions)
    net_dollar_gamma = sum(p["dollar_gamma"] for p in positions)
    net_dollar_vega = sum(p["dollar_vega"] for p in positions)
    net_dollar_theta = sum(p["dollar_theta"] for p in positions)
    total_MtM = sum(p["MtM"] for p in positions)
    daily_PnL = sum(p["unrealized_PnL"] for p in positions)

    # Simple attribution breakdown
    theta_pnl = net_dollar_theta  # overnight theta decay
    # vega and delta attribution approximated from Greeks contribution
    vega_pnl = net_dollar_vega  # placeholder: 1 vol-pt equivalent
    delta_pnl = daily_PnL - theta_pnl - vega_pnl

    return {
        "net_dollar_delta": round(net_dollar_delta, 2),
        "net_dollar_gamma": round(net_dollar_gamma, 4),
        "net_dollar_vega": round(net_dollar_vega, 2),
        "net_dollar_theta": round(net_dollar_theta, 2),
        "total_MtM": round(total_MtM, 2),
        "daily_PnL": round(daily_PnL, 2),
        "PnL_attribution": {
            "delta_pnl": round(delta_pnl, 2),
            "vega_pnl": round(vega_pnl, 2),
            "theta_pnl": round(theta_pnl, 2),
        },
        "positions": positions,
        "position_count": len(positions),
    }


# ---------------------------------------------------------------------------
# Report payload builder
# ---------------------------------------------------------------------------

def build_report_payload(
    portfolio_metrics: dict,
    market_context: dict,
    prior_snapshot: Optional[dict],
    alerts: list[str],
) -> dict:
    """
    Assemble the complete JSON payload passed to Claude and embedded in the
    PDF report.

    Parameters
    ----------
    portfolio_metrics : dict
        Output of :func:`aggregate_portfolio`.
    market_context : dict
        Output of ``data.market_data.get_market_context``.
    prior_snapshot : dict or None
        Yesterday's snapshot from the database.  When ``None`` (first run),
        prior Greeks are reported as 0.
    alerts : list[str]
        Human-readable alert strings (limit breaches, expiry warnings, etc.).

    Returns
    -------
    dict
        Fully structured payload ready for JSON serialisation.
    """
    prior = prior_snapshot or {}
    prior_agg = prior.get("portfolio_metrics", {})

    payload = {
        "report_date": datetime.today().strftime("%Y-%m-%d"),
        "portfolio_metrics": {
            "current": {
                "net_dollar_delta": portfolio_metrics.get("net_dollar_delta", 0),
                "net_dollar_gamma": portfolio_metrics.get("net_dollar_gamma", 0),
                "net_dollar_vega": portfolio_metrics.get("net_dollar_vega", 0),
                "net_dollar_theta": portfolio_metrics.get("net_dollar_theta", 0),
                "total_MtM": portfolio_metrics.get("total_MtM", 0),
                "daily_PnL": portfolio_metrics.get("daily_PnL", 0),
            },
            "prior": {
                "net_dollar_delta": prior_agg.get("current", {}).get("net_dollar_delta", 0),
                "net_dollar_gamma": prior_agg.get("current", {}).get("net_dollar_gamma", 0),
                "net_dollar_vega": prior_agg.get("current", {}).get("net_dollar_vega", 0),
                "net_dollar_theta": prior_agg.get("current", {}).get("net_dollar_theta", 0),
                "total_MtM": prior_agg.get("current", {}).get("total_MtM", 0),
                "daily_PnL": 0,  # yesterday's intraday P&L is not additive
            },
            "PnL_attribution": portfolio_metrics.get("PnL_attribution", {}),
        },
        "positions": portfolio_metrics.get("positions", []),
        "market_context": market_context,
        "alerts": alerts,
        "is_first_run": prior_snapshot is None,
    }
    return payload
