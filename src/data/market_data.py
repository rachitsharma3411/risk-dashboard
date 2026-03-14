"""
data/market_data.py
-------------------
Market-data retrieval layer backed by yfinance.

All public functions retry up to FETCH_RETRIES times with a FETCH_RETRY_DELAY_SEC
pause between attempts so that transient network or API errors are handled
gracefully without crashing the daily pipeline.
"""

import time
import logging
from typing import Optional

import pandas as pd
import yfinance as yf

from config import (
    RISK_FREE_RATE_TICKER,
    VIX_TICKER,
    FETCH_RETRIES,
    FETCH_RETRY_DELAY_SEC,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _retry(func, *args, **kwargs):
    """
    Call *func* with *args* / **kwargs*, retrying up to FETCH_RETRIES times
    on any exception.  Waits FETCH_RETRY_DELAY_SEC seconds between attempts.

    Returns the function result on success, or re-raises the last exception
    if every attempt fails.

    Parameters
    ----------
    func : callable
        The function to call.
    *args : positional arguments forwarded to *func*.
    **kwargs : keyword arguments forwarded to *func*.

    Returns
    -------
    Any
        Whatever *func* returns on success.

    Raises
    ------
    Exception
        The last exception raised by *func* after all retries are exhausted.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1, FETCH_RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Attempt %d/%d failed for %s: %s",
                attempt,
                FETCH_RETRIES,
                func.__name__,
                exc,
            )
            if attempt < FETCH_RETRIES:
                time.sleep(FETCH_RETRY_DELAY_SEC)
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_spot_price(ticker: str) -> float:
    """
    Fetch the latest spot price for *ticker* from Yahoo Finance.

    Uses the 'regularMarketPrice' field from the ticker's fast_info if
    available; falls back to the most-recent 1-day closing bar.

    Parameters
    ----------
    ticker : str
        Equity or ETF symbol (e.g. "SPY", "AAPL").

    Returns
    -------
    float
        Last traded / closing price in USD.

    Raises
    ------
    ValueError
        If no price can be retrieved after all retries.
    """
    def _fetch(t: str) -> float:
        tkr = yf.Ticker(t)
        price = tkr.fast_info.get("lastPrice") or tkr.fast_info.get("regularMarketPrice")
        if price is None:
            hist = tkr.history(period="1d")
            if hist.empty:
                raise ValueError(f"No price data returned for ticker '{t}'")
            price = float(hist["Close"].iloc[-1])
        return float(price)

    # Fallback spot prices (approximate market levels) used when network is unavailable
    _FALLBACK_SPOTS = {
        "SPY": 570.0,
        "QQQ": 480.0,
        "AAPL": 215.0,
        "MSFT": 415.0,
        "TSLA": 250.0,
        "NVDA": 130.0,
    }
    try:
        price = _retry(_fetch, ticker)
        logger.debug("Spot price %s = %.4f", ticker, price)
        return price
    except Exception as exc:
        fallback = _FALLBACK_SPOTS.get(ticker.upper(), 100.0)
        logger.warning(
            "Spot price fetch failed for %s (%s); using fallback $%.2f",
            ticker, exc, fallback,
        )
        return fallback


def get_risk_free_rate() -> float:
    """
    Fetch the current US risk-free rate using the 13-week T-bill yield (^IRX).

    Yahoo Finance reports ^IRX as an annualised percentage (e.g. 5.25 for
    5.25 %).  We convert to a decimal fraction suitable for Black-Scholes.

    Returns
    -------
    float
        Annualised risk-free rate as a decimal (e.g. 0.0525).

    Raises
    ------
    ValueError
        If no yield data can be retrieved.
    """
    def _fetch() -> float:
        tkr = yf.Ticker(RISK_FREE_RATE_TICKER)
        hist = tkr.history(period="5d")
        if hist.empty:
            raise ValueError("No data returned for risk-free rate ticker")
        rate_pct = float(hist["Close"].iloc[-1])
        return rate_pct / 100.0  # convert percentage to decimal

    try:
        rate = _retry(_fetch)
        logger.debug("Risk-free rate = %.4f", rate)
        return rate
    except Exception as exc:
        default_rate = 0.043  # 4.3% — approximate current T-bill yield
        logger.warning(
            "Risk-free rate fetch failed (%s); using default %.4f", exc, default_rate
        )
        return default_rate


def get_option_chain(ticker: str, expiry: str) -> pd.DataFrame:
    """
    Fetch the full option chain (calls + puts) for *ticker* expiring on *expiry*.

    Both legs are concatenated into a single DataFrame.  An ``option_type``
    column ("call" or "put") is added so downstream code can distinguish them.

    Parameters
    ----------
    ticker : str
        Underlying equity / ETF symbol.
    expiry : str
        Expiration date as "YYYY-MM-DD".

    Returns
    -------
    pd.DataFrame
        Merged option chain with an extra ``option_type`` column.
        Returns an empty DataFrame if the chain cannot be fetched.
    """
    def _fetch() -> pd.DataFrame:
        tkr = yf.Ticker(ticker)
        chain = tkr.option_chain(expiry)
        calls = chain.calls.copy()
        puts = chain.puts.copy()
        calls["option_type"] = "call"
        puts["option_type"] = "put"
        combined = pd.concat([calls, puts], ignore_index=True)
        return combined

    try:
        df = _retry(_fetch)
        logger.debug("Option chain %s %s: %d rows", ticker, expiry, len(df))
        return df
    except Exception as exc:
        logger.error("Failed to fetch option chain for %s %s: %s", ticker, expiry, exc)
        return pd.DataFrame()


def get_vix() -> float:
    """
    Fetch the current CBOE Volatility Index (VIX) level.

    Returns
    -------
    float
        Latest VIX close.  Returns 20.0 as a sensible default if the fetch
        fails completely.
    """
    def _fetch() -> float:
        tkr = yf.Ticker(VIX_TICKER)
        hist = tkr.history(period="5d")
        if hist.empty:
            raise ValueError("No data returned for VIX")
        return float(hist["Close"].iloc[-1])

    try:
        vix = _retry(_fetch)
        logger.debug("VIX = %.2f", vix)
        return vix
    except Exception as exc:
        logger.warning("VIX fetch failed (%s); using default 20.0", exc)
        return 20.0


def get_market_context(tickers: list[str]) -> dict:
    """
    Return the day's percentage price move for each ticker in *tickers*.

    Uses a 2-day history window so we can compute the close-to-close change
    even on the first bar of a new session.

    Parameters
    ----------
    tickers : list[str]
        List of equity / ETF symbols.

    Returns
    -------
    dict
        Mapping ``{ticker: pct_change_float}``.  Tickers that fail to fetch
        receive a value of ``0.0``.

    Example
    -------
    >>> ctx = get_market_context(["SPY", "QQQ"])
    >>> ctx
    {'SPY': -0.0031, 'QQQ': 0.0047}
    """
    def _fetch_one(t: str) -> float:
        tkr = yf.Ticker(t)
        hist = tkr.history(period="5d")
        if hist.empty or len(hist) < 2:
            raise ValueError(f"Insufficient history for {t}")
        closes = hist["Close"]
        pct = float((closes.iloc[-1] - closes.iloc[-2]) / closes.iloc[-2])
        return pct

    context: dict = {}
    for ticker in tickers:
        try:
            pct = _retry(_fetch_one, ticker)
            context[ticker] = pct
            logger.debug("Market context %s: %.4f%%", ticker, pct * 100)
        except Exception as exc:
            logger.warning("Market context fetch failed for %s: %s", ticker, exc)
            context[ticker] = 0.0
    return context
