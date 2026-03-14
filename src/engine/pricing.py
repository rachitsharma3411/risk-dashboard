"""
engine/pricing.py
-----------------
Closed-form options pricing using the Black-Scholes-Merton model and
implied-volatility solving via Newton-Raphson with a Brent-method fallback.
"""

import logging
import math
from typing import Literal

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

logger = logging.getLogger(__name__)

OptionType = Literal["call", "put"]

# ---------------------------------------------------------------------------
# Black-Scholes
# ---------------------------------------------------------------------------

def black_scholes(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType,
) -> float:
    """
    Compute the Black-Scholes-Merton theoretical price of a European option.

    Parameters
    ----------
    S : float
        Current spot price of the underlying asset.
    K : float
        Option strike price.
    T : float
        Time to expiration in years (must be > 0).
    r : float
        Annualised continuously-compounded risk-free rate (decimal, e.g. 0.05).
    sigma : float
        Annualised implied volatility (decimal, e.g. 0.25).
    option_type : {"call", "put"}
        Option flavour.

    Returns
    -------
    float
        Theoretical option price.  Returns the intrinsic value if T <= 0 or
        sigma <= 0 to avoid division-by-zero.

    Examples
    --------
    >>> black_scholes(100, 100, 1.0, 0.05, 0.20, "call")
    10.4506...
    """
    if T <= 0 or sigma <= 0:
        return intrinsic_value(S, K, option_type)

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return max(float(price), 0.0)


# ---------------------------------------------------------------------------
# Implied volatility
# ---------------------------------------------------------------------------

def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """
    Solve for the implied volatility that makes the Black-Scholes price equal
    to *market_price*.

    Algorithm
    ---------
    1. Newton-Raphson from an initial guess of σ = 0.20 (20 %).
    2. If Newton-Raphson diverges or fails to converge, fall back to
       Brent's method on the interval [1e-6, 5.0].
    3. If both methods fail, return 0.20 as a safe default.

    Parameters
    ----------
    market_price : float
        Observed mid-market option price.
    S : float
        Current spot price.
    K : float
        Strike price.
    T : float
        Time to expiration in years.
    r : float
        Annualised risk-free rate (decimal).
    option_type : {"call", "put"}
        Option flavour.
    tol : float, optional
        Convergence tolerance on the price residual.  Default 1e-6.
    max_iter : int, optional
        Maximum Newton-Raphson iterations.  Default 100.

    Returns
    -------
    float
        Implied volatility as a decimal (e.g. 0.25 for 25 %).  Falls back to
        0.20 if solving fails.
    """
    DEFAULT_IV = 0.20

    # Edge-case: if market_price is at or below intrinsic we cannot solve
    intrinsic = intrinsic_value(S, K, option_type)
    if market_price <= 0 or T <= 0:
        return DEFAULT_IV
    if market_price < intrinsic:
        market_price = intrinsic + 1e-4  # nudge above parity

    # -----------------------------------------------------------------------
    # Newton-Raphson
    # -----------------------------------------------------------------------
    sigma = DEFAULT_IV
    for i in range(max_iter):
        price = black_scholes(S, K, T, r, sigma, option_type)
        # vega for Newton step
        if T <= 0 or sigma <= 0:
            break
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        vega_raw = S * norm.pdf(d1) * math.sqrt(T)  # price-space vega

        residual = price - market_price
        if abs(residual) < tol:
            logger.debug("IV Newton-Raphson converged in %d iterations: σ=%.6f", i + 1, sigma)
            return float(np.clip(sigma, 1e-6, 5.0))

        if abs(vega_raw) < 1e-10:
            break  # gradient too flat — switch to Brent

        sigma -= residual / vega_raw
        if sigma <= 0:
            sigma = DEFAULT_IV  # reset if step overshoots

    # -----------------------------------------------------------------------
    # Brent fallback
    # -----------------------------------------------------------------------
    try:
        def objective(sig: float) -> float:
            return black_scholes(S, K, T, r, sig, option_type) - market_price

        # Check bracket validity
        lo, hi = 1e-6, 5.0
        f_lo = objective(lo)
        f_hi = objective(hi)
        if f_lo * f_hi > 0:
            logger.debug("Brent bracket invalid for IV; returning default %.2f", DEFAULT_IV)
            return DEFAULT_IV

        iv = brentq(objective, lo, hi, xtol=tol, maxiter=500)
        logger.debug("IV Brent converged: σ=%.6f", iv)
        return float(np.clip(iv, 1e-6, 5.0))
    except Exception as exc:
        logger.warning("IV solving failed entirely (%s); returning default %.2f", exc, DEFAULT_IV)
        return DEFAULT_IV


# ---------------------------------------------------------------------------
# Intrinsic value
# ---------------------------------------------------------------------------

def intrinsic_value(S: float, K: float, option_type: OptionType) -> float:
    """
    Compute the intrinsic (exercise) value of an option.

    Parameters
    ----------
    S : float
        Current spot price.
    K : float
        Strike price.
    option_type : {"call", "put"}
        Option flavour.

    Returns
    -------
    float
        max(S - K, 0) for calls; max(K - S, 0) for puts.

    Examples
    --------
    >>> intrinsic_value(105, 100, "call")
    5.0
    >>> intrinsic_value(105, 100, "put")
    0.0
    """
    if option_type == "call":
        return max(S - K, 0.0)
    return max(K - S, 0.0)
