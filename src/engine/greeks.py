"""
engine/greeks.py
----------------
Black-Scholes analytical Greeks for European options.

All functions return sensible zero-values when T <= 0 or sigma <= 0 to
avoid numerical blow-ups near expiration.

Convention notes
----------------
- *vega* is scaled to represent the price change per **1 percentage-point**
  move in implied volatility (i.e. raw_vega / 100).
- *theta* is scaled to represent the price change per **calendar day**
  (i.e. raw_theta / 365).
- *rho* is the price change per **1 percentage-point** move in the risk-free
  rate (i.e. raw_rho / 100).
"""

import logging
import math
from typing import Literal

from scipy.stats import norm

logger = logging.getLogger(__name__)

OptionType = Literal["call", "put"]


def _d1_d2(
    S: float, K: float, T: float, r: float, sigma: float
) -> tuple[float, float]:
    """
    Compute the d1 and d2 intermediate values shared by all BS Greeks.

    Parameters
    ----------
    S, K, T, r, sigma : float
        Standard Black-Scholes inputs.

    Returns
    -------
    tuple[float, float]
        (d1, d2)
    """
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------

def delta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType,
) -> float:
    """
    Rate of change of option price with respect to the underlying spot price.

    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to expiration in years.
    r : float
        Annualised risk-free rate (decimal).
    sigma : float
        Annualised implied volatility (decimal).
    option_type : {"call", "put"}
        Option flavour.

    Returns
    -------
    float
        Delta in the range (-1, 0) for puts and (0, 1) for calls.
        Returns +1 / -1 for deep in-the-money options at expiry.
    """
    if T <= 0 or sigma <= 0:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0

    d1, _ = _d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return float(norm.cdf(d1))
    return float(norm.cdf(d1) - 1.0)


def gamma(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> float:
    """
    Rate of change of delta with respect to the underlying spot price.

    Gamma is identical for calls and puts (put-call parity).

    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to expiration in years.
    r : float
        Annualised risk-free rate (decimal).
    sigma : float
        Annualised implied volatility (decimal).

    Returns
    -------
    float
        Gamma (always >= 0).
    """
    if T <= 0 or sigma <= 0:
        return 0.0

    d1, _ = _d1_d2(S, K, T, r, sigma)
    return float(norm.pdf(d1) / (S * sigma * math.sqrt(T)))


def vega(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> float:
    """
    Sensitivity of the option price to a **1-percentage-point** change in
    implied volatility.

    The raw Black-Scholes vega is divided by 100 so the returned value
    represents: Δprice / Δσ(1 pp).

    Vega is identical for calls and puts.

    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to expiration in years.
    r : float
        Annualised risk-free rate (decimal).
    sigma : float
        Annualised implied volatility (decimal).

    Returns
    -------
    float
        Vega per 1 vol point (always >= 0).
    """
    if T <= 0 or sigma <= 0:
        return 0.0

    d1, _ = _d1_d2(S, K, T, r, sigma)
    raw_vega = S * norm.pdf(d1) * math.sqrt(T)
    return float(raw_vega / 100.0)


def theta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType,
) -> float:
    """
    Daily time-decay of the option price (per calendar day).

    The raw Black-Scholes theta (annualised) is divided by 365 so the
    returned value represents the expected price change overnight.

    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to expiration in years.
    r : float
        Annualised risk-free rate (decimal).
    sigma : float
        Annualised implied volatility (decimal).
    option_type : {"call", "put"}
        Option flavour.

    Returns
    -------
    float
        Theta per calendar day (typically negative — options lose value over
        time).
    """
    if T <= 0 or sigma <= 0:
        return 0.0

    d1, d2 = _d1_d2(S, K, T, r, sigma)
    common = -(S * norm.pdf(d1) * sigma) / (2.0 * math.sqrt(T))

    if option_type == "call":
        raw_theta = common - r * K * math.exp(-r * T) * norm.cdf(d2)
    else:
        raw_theta = common + r * K * math.exp(-r * T) * norm.cdf(-d2)

    return float(raw_theta / 365.0)


def rho(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType,
) -> float:
    """
    Sensitivity of the option price to a **1-percentage-point** change in the
    risk-free interest rate.

    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to expiration in years.
    r : float
        Annualised risk-free rate (decimal).
    sigma : float
        Annualised implied volatility (decimal).
    option_type : {"call", "put"}
        Option flavour.

    Returns
    -------
    float
        Rho per 1 percentage-point rate move.
    """
    if T <= 0 or sigma <= 0:
        return 0.0

    _, d2 = _d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        raw_rho = K * T * math.exp(-r * T) * norm.cdf(d2)
    else:
        raw_rho = -K * T * math.exp(-r * T) * norm.cdf(-d2)

    return float(raw_rho / 100.0)
