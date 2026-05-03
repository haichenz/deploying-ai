"""Quant utilities for Vega: options pricing, Greeks, scenario analysis, and statistics.

All functions are pure (no I/O, no global state) and JSON-serializable in their returns.
Designed for an intraday SPX short put spread strategy:
- Spread notation: short_strike/long_strike where short > long
- Default width: 5 points (override-able)
- Multiplier: $100/contract (SPX index options)
- DTE in minutes (converted to years internally for Black-Scholes)
- IV as decimal (0.15 not 15)
- Risk-free rate as decimal (0.05 not 5)
"""

import math
from typing import Optional
import numpy as np
from scipy.stats import norm


MULTIPLIER = 100.0  # SPX index option contract multiplier
MINUTES_PER_YEAR = 252 * 6.5 * 60  # 252 trading days * 6.5 hours/day * 60 min


def _bs_put(S: float, K: float, T: float, r: float, q: float, sigma: float) -> dict:
    """Internal: Black-Scholes-Merton put price + Greeks. T in years."""
    if T <= 0 or sigma <= 0:
        intrinsic = max(K - S, 0.0)
        return {
            "price": round(intrinsic, 4),
            "delta": -1.0 if S < K else 0.0,
            "gamma": 0.0,
            "theta_per_day": 0.0,
            "vega": 0.0,
        }

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    discount_K = K * math.exp(-r * T)
    discount_S = S * math.exp(-q * T)

    price = discount_K * norm.cdf(-d2) - discount_S * norm.cdf(-d1)
    delta = -math.exp(-q * T) * norm.cdf(-d1)
    gamma = math.exp(-q * T) * norm.pdf(d1) / (S * sigma * sqrt_T)
    theta_year = (
        -discount_S * norm.pdf(d1) * sigma / (2 * sqrt_T)
        + r * discount_K * norm.cdf(-d2)
        - q * discount_S * norm.cdf(-d1)
    )
    theta_per_day = theta_year / 365.0
    vega = discount_S * norm.pdf(d1) * sqrt_T / 100.0  # per 1% vol move

    return {
        "price": round(price, 4),
        "delta": round(delta, 6),
        "gamma": round(gamma, 6),
        "theta_per_day": round(theta_per_day, 6),
        "vega": round(vega, 6),
    }


def price_short_put_spread(
    spot: float,
    short_K: float,
    long_K: float,
    dte_minutes: float,
    sigma_short: float,
    sigma_long: float,
    r: float = 0.05,
    q: float = 0.0,
) -> dict:
    """
    Price a short put credit spread under Black-Scholes-Merton.

    Args:
        spot: underlying spot price (SPX level)
        short_K: short leg strike (sold put, higher strike)
        long_K: long leg strike (bought put, lower strike)
        dte_minutes: time to expiration in minutes
        sigma_short: implied vol on short leg (decimal, e.g., 0.15)
        sigma_long: implied vol on long leg (decimal)
        r: risk-free rate (decimal, default 0.05)
        q: continuous dividend yield (decimal, default 0.0)

    Returns:
        credit, max_loss, max_loss_dollars, breakeven, width_points,
        net Greeks, and per-leg detail dicts.
    """
    if short_K <= long_K:
        raise ValueError(f"short_K ({short_K}) must be > long_K ({long_K}) for a put credit spread")
    if dte_minutes <= 0:
        raise ValueError(f"dte_minutes must be positive, got {dte_minutes}")

    T = dte_minutes / MINUTES_PER_YEAR

    short_leg = _bs_put(spot, short_K, T, r, q, sigma_short)
    long_leg = _bs_put(spot, long_K, T, r, q, sigma_long)

    credit = short_leg["price"] - long_leg["price"]
    width = short_K - long_K
    max_loss = width - credit

    return {
        "credit": round(credit, 4),
        "max_loss": round(max_loss, 4),
        "max_loss_dollars": round(max_loss * MULTIPLIER, 2),
        "breakeven": round(short_K - credit, 4),
        "width_points": round(width, 4),
        "net_delta": round(-short_leg["delta"] + long_leg["delta"], 6),
        "net_gamma": round(-short_leg["gamma"] + long_leg["gamma"], 6),
        "net_theta_per_day": round(-short_leg["theta_per_day"] + long_leg["theta_per_day"], 6),
        "net_vega": round(-short_leg["vega"] + long_leg["vega"], 6),
        "short_leg": short_leg,
        "long_leg": long_leg,
    }


def spread_pnl_at_spot(
    short_K: float,
    long_K: float,
    credit_received: float,
    spot_now: float,
    dte_min_remaining: float,
    sigma_short: float,
    sigma_long: float,
    r: float = 0.05,
    q: float = 0.0,
) -> dict:
    """
    Compute mark-to-market P&L on an open short put spread at a hypothetical spot level.

    Args:
        short_K, long_K: spread strikes (short > long)
        credit_received: credit collected at entry, in dollars per spread (e.g., 1.10)
        spot_now: hypothetical spot level for the scenario
        dte_min_remaining: minutes until expiration at scenario time
        sigma_short, sigma_long: IVs at scenario time
        r, q: rates

    Returns:
        spread_value_now, pnl_per_contract, otm_points, moneyness,
        credit_captured_pct, max_loss, breakeven.
    """
    pricing = price_short_put_spread(
        spot=spot_now, short_K=short_K, long_K=long_K,
        dte_minutes=dte_min_remaining,
        sigma_short=sigma_short, sigma_long=sigma_long,
        r=r, q=q,
    )
    spread_value = pricing["credit"]
    pnl_per_contract = (credit_received - spread_value) * MULTIPLIER
    otm_points = spot_now - short_K

    if otm_points > 1.0:
        moneyness = "OTM"
    elif otm_points < -1.0:
        moneyness = "ITM"
    else:
        moneyness = "ATM"

    captured = 1.0 - (spread_value / credit_received) if credit_received > 0 else 0.0

    return {
        "spread_value_now": round(spread_value, 4),
        "pnl_per_contract": round(pnl_per_contract, 2),
        "otm_points": round(otm_points, 2),
        "moneyness": moneyness,
        "credit_captured_pct": round(captured, 4),
        "max_loss": pricing["max_loss"],
        "breakeven": pricing["breakeven"],
    }


def compute_returns_stats(
    daily_pnl: list[float],
    capital: Optional[float] = None,
    periods_per_year: int = 252,
) -> dict:
    """
    Compute summary statistics on a daily P&L series.

    Args:
        daily_pnl: list of daily P&L values in dollars (one entry per trading day).
                   Days with no trades should be 0.0, not missing.
        capital: optional reference capital for percentage-return view.
        periods_per_year: 252 for daily.

    Returns:
        n_days, total_pnl, mean/std, Sharpe, Sortino, drawdown, best/worst day,
        and optional annualized return/vol if capital provided.
    """
    if not daily_pnl:
        raise ValueError("daily_pnl is empty")

    pnl = np.asarray(daily_pnl, dtype=float)
    n = len(pnl)
    total = float(pnl.sum())
    mean = float(pnl.mean())
    std = float(pnl.std(ddof=1)) if n > 1 else 0.0

    sharpe = (mean / std) * math.sqrt(periods_per_year) if std > 0 else 0.0

    downside = pnl[pnl < 0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sortino = (mean / downside_std) * math.sqrt(periods_per_year) if downside_std > 0 else 0.0

    cum = np.cumsum(pnl)
    running_peak = np.maximum.accumulate(cum)
    drawdown = cum - running_peak
    max_dd = float(drawdown.min())

    in_dd = drawdown < 0
    max_run = current_run = 0
    for x in in_dd:
        if x:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0

    out = {
        "n_days": n,
        "total_pnl": round(total, 2),
        "mean_daily_pnl": round(mean, 4),
        "std_daily_pnl": round(std, 4),
        "sharpe_annualized": round(sharpe, 4),
        "sortino_annualized": round(sortino, 4),
        "max_drawdown_dollars": round(max_dd, 2),
        "max_drawdown_pct": None,
        "max_dd_duration_days": int(max_run),
        "best_day": round(float(pnl.max()), 2),
        "worst_day": round(float(pnl.min()), 2),
        "annualized_return_pct": None,
        "annualized_vol_pct": None,
    }

    if capital is not None and capital > 0:
        out["max_drawdown_pct"] = round(max_dd / capital, 6)
        out["annualized_return_pct"] = round((mean * periods_per_year) / capital, 6)
        out["annualized_vol_pct"] = round((std * math.sqrt(periods_per_year)) / capital, 6)

    return out


def compute_trade_stats(per_trade_pnl: list[float]) -> dict:
    """
    Compute trade-level statistics.

    Args:
        per_trade_pnl: list of P&L values, one per trade.

    Returns:
        n_trades, n_wins, n_losses, n_scratches, hit_rate, avg_win, avg_loss,
        win_loss_ratio, profit_factor, expectancy_per_trade, largest win/loss, total_pnl.
    """
    if not per_trade_pnl:
        raise ValueError("per_trade_pnl is empty")

    arr = np.asarray(per_trade_pnl, dtype=float)
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    scratches = arr[arr == 0]

    n_wins = len(wins)
    n_losses = len(losses)
    decided = n_wins + n_losses

    avg_win = float(wins.mean()) if n_wins > 0 else 0.0
    avg_loss = float(abs(losses.mean())) if n_losses > 0 else 0.0
    sum_wins = float(wins.sum()) if n_wins > 0 else 0.0
    sum_losses = float(abs(losses.sum())) if n_losses > 0 else 0.0

    return {
        "n_trades": len(arr),
        "n_wins": n_wins,
        "n_losses": n_losses,
        "n_scratches": len(scratches),
        "hit_rate": round(n_wins / decided, 4) if decided > 0 else 0.0,
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "win_loss_ratio": round(avg_win / avg_loss, 4) if avg_loss > 0 else float("inf"),
        "profit_factor": round(sum_wins / sum_losses, 4) if sum_losses > 0 else float("inf"),
        "expectancy_per_trade": round(float(arr.mean()), 4),
        "largest_win": round(float(arr.max()), 2),
        "largest_loss": round(float(arr.min()), 2),
        "total_pnl": round(float(arr.sum()), 2),
    }


def mae_mfe(pnl_trajectory: list[float]) -> dict:
    """
    Compute MAE / MFE statistics on a trade's intraday P&L path.

    Args:
        pnl_trajectory: ordered list of P&L values during a trade's life.
                        Last value is treated as the closing P&L.

    Returns:
        final_pnl, mae, mfe, mae_minute, mfe_minute, trade_duration,
        efficiency, hold_time_to_mfe, give_back_after_mfe.
    """
    if not pnl_trajectory:
        raise ValueError("pnl_trajectory is empty")

    arr = np.asarray(pnl_trajectory, dtype=float)
    final = float(arr[-1])
    mae = float(arr.min())
    mfe = float(arr.max())
    mae_idx = int(arr.argmin())
    mfe_idx = int(arr.argmax())

    efficiency = None
    if final > 0 and mfe > 0:
        efficiency = round(final / mfe, 4)

    return {
        "final_pnl": round(final, 2),
        "mae": round(mae, 2),
        "mfe": round(mfe, 2),
        "mae_minute": mae_idx,
        "mfe_minute": mfe_idx,
        "trade_duration": len(arr),
        "efficiency": efficiency,
        "hold_time_to_mfe": mfe_idx,
        "give_back_after_mfe": round(mfe - final, 2),
    }


def kelly_position_size(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    account_equity: float,
    max_loss_per_contract: float,
    kelly_multiplier: float = 0.25,
) -> dict:
    """
    Compute Kelly-based position sizing for a binary-outcome strategy.

    Args:
        win_rate: probability of a winning trade (0-1)
        avg_win: average win size in dollars per contract
        avg_loss: average loss size in dollars per contract (positive number)
        account_equity: total account equity in dollars
        max_loss_per_contract: capital at risk per contract
        kelly_multiplier: fraction of full Kelly to use (default 0.25 = quarter Kelly)

    Returns:
        kelly_fraction_full, kelly_fraction_used, edge_per_dollar,
        recommended_capital_at_risk, recommended_contracts at full and fractional Kelly,
        and optional warning string.
    """
    if not (0.0 <= win_rate <= 1.0):
        raise ValueError(f"win_rate must be in [0,1], got {win_rate}")
    if avg_win <= 0 or avg_loss <= 0:
        raise ValueError("avg_win and avg_loss must both be positive")
    if account_equity <= 0 or max_loss_per_contract <= 0:
        raise ValueError("account_equity and max_loss_per_contract must be positive")

    b = avg_win / avg_loss
    p = win_rate
    q = 1.0 - p
    f_full = p - q / b
    f_used = f_full * kelly_multiplier

    edge_per_dollar = (p * avg_win - q * avg_loss) / avg_loss

    warning = None
    if f_full <= 0:
        warning = "No edge — Kelly is non-positive. Strategy expectancy is negative or zero."
        return {
            "kelly_fraction_full": round(f_full, 4),
            "kelly_fraction_used": 0.0,
            "edge_per_dollar": round(edge_per_dollar, 4),
            "recommended_capital_at_risk": 0.0,
            "recommended_contracts_full_kelly": 0,
            "recommended_contracts_fractional": 0,
            "warning": warning,
        }

    if f_full > 0.25:
        warning = "Full Kelly exceeds 25% of equity — consider lower fraction for real-money trading."

    capital_at_risk_full = account_equity * f_full
    capital_at_risk_used = account_equity * f_used

    contracts_full = int(capital_at_risk_full // max_loss_per_contract)
    contracts_used = int(capital_at_risk_used // max_loss_per_contract)

    return {
        "kelly_fraction_full": round(f_full, 4),
        "kelly_fraction_used": round(f_used, 4),
        "edge_per_dollar": round(edge_per_dollar, 4),
        "recommended_capital_at_risk": round(capital_at_risk_used, 2),
        "recommended_contracts_full_kelly": contracts_full,
        "recommended_contracts_fractional": contracts_used,
        "warning": warning,
    }
