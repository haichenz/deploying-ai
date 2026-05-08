"""Live market regime snapshot via Yahoo Finance JSON. [Phase 4 — Service 1]

Fetches SPX, the VIX complex (VIX9D / VIX / VIX3M), and 10Y yield, then
synthesizes a regime classification with explicit relevance to short premium
strategies (the user's intraday SPX put credit spread).

Pure functional design with one network-touching public function. Classifiers
are exposed for testing.
"""

from datetime import datetime, timezone
from typing import Optional

import requests

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

_YAHOO_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"
_HEADERS = {
    "User-Agent": "Mozilla/5.0",
}
_TIMEOUT = 5.0  # seconds per symbol


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def get_market_regime() -> dict:
    """
    Fetch a synthesized live market regime snapshot.

    Returns:
        {
            "spx":      {price, prior_close, day_change_pct, five_day_range, five_day_change_pct} | None,
            "vix":      {level, five_day_change_pct, regime} | None,
            "vix_term": {vix9d, vix, vix3m, structure, signal, regime_for_short_premium} | None,
            "tnx":      {yield_pct, five_day_change_bps} | None,
            "timestamp_utc": ISO 8601 string,
            "data_quality": {symbol: "ok" | "unavailable" | "partial"},
        }

    Any field can be None if its underlying fetch failed. Callers should
    handle missing fields explicitly. Vega should narrate honestly when
    a field is None — never fabricate.
    """
    quality: dict[str, str] = {}

    spx_block, spx_q = _build_spx_block()
    quality["spx"] = spx_q

    vix_block, vix_q, vix_level = _build_vix_block()
    quality["vix"] = vix_q

    vix_term_block, vix_term_q = _build_vix_term_block(vix_level)
    quality["vix_term"] = vix_term_q

    tnx_block, tnx_q = _build_tnx_block()
    quality["tnx"] = tnx_q

    return {
        "spx": spx_block,
        "vix": vix_block,
        "vix_term": vix_term_block,
        "tnx": tnx_block,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_quality": quality,
    }


# --------------------------------------------------------------------------- #
# Block builders (one per symbol/group)
# --------------------------------------------------------------------------- #

def _build_spx_block() -> tuple[Optional[dict], str]:
    chart = _fetch_chart("^GSPC")
    if chart is None:
        return None, "unavailable"

    spot = _meta_price(chart)
    closes = _closes_from_chart(chart)
    prior_close = _meta_prior_close(chart) or (closes[-2] if len(closes) >= 2 else None)

    if spot is None or prior_close is None or not closes:
        return None, "partial"

    day_change_pct = (spot - prior_close) / prior_close * 100
    five_day_change_pct = (
        (spot - closes[0]) / closes[0] * 100 if len(closes) >= 2 else None
    )

    return {
        "price": round(spot, 2),
        "prior_close": round(prior_close, 2),
        "day_change_pct": round(day_change_pct, 3),
        "five_day_range": [round(min(closes), 2), round(max(closes), 2)],
        "five_day_change_pct": round(five_day_change_pct, 3) if five_day_change_pct is not None else None,
    }, "ok"


def _build_vix_block() -> tuple[Optional[dict], str, Optional[float]]:
    chart = _fetch_chart("^VIX")
    if chart is None:
        return None, "unavailable", None

    level = _meta_price(chart) or _last_close(chart)
    closes = _closes_from_chart(chart)

    if level is None:
        return None, "partial", None

    five_day_change_pct = (
        (level - closes[0]) / closes[0] * 100 if len(closes) >= 2 else None
    )

    return {
        "level": round(level, 2),
        "five_day_change_pct": round(five_day_change_pct, 2) if five_day_change_pct is not None else None,
        "regime": classify_vix_regime(level),
    }, "ok", level


def _build_vix_term_block(vix_level: Optional[float]) -> tuple[Optional[dict], str]:
    vix9d_chart = _fetch_chart("^VIX9D")
    vix3m_chart = _fetch_chart("^VIX3M")

    vix9d = _meta_price(vix9d_chart) or _last_close(vix9d_chart) if vix9d_chart else None
    vix3m = _meta_price(vix3m_chart) or _last_close(vix3m_chart) if vix3m_chart else None

    if vix_level is None or vix9d is None or vix3m is None:
        return None, "unavailable" if (vix9d is None and vix3m is None) else "partial"

    classification = classify_term_structure(vix9d, vix_level, vix3m)
    return {
        "vix9d": round(vix9d, 2),
        "vix": round(vix_level, 2),
        "vix3m": round(vix3m, 2),
        "structure": classification["structure"],
        "signal": classification["signal"],
        "regime_for_short_premium": classification["regime"],
    }, "ok"


def _build_tnx_block() -> tuple[Optional[dict], str]:
    chart = _fetch_chart("^TNX")
    if chart is None:
        return None, "unavailable"

    yield_pct = _meta_price(chart) or _last_close(chart)
    closes = _closes_from_chart(chart)

    if yield_pct is None:
        return None, "partial"

    five_day_change_bps = (
        (yield_pct - closes[0]) * 100 if len(closes) >= 2 else None
    )

    return {
        "yield_pct": round(yield_pct, 3),
        "five_day_change_bps": round(five_day_change_bps, 1) if five_day_change_bps is not None else None,
    }, "ok"


# --------------------------------------------------------------------------- #
# Classifiers (pure functions — easy to unit test)
# --------------------------------------------------------------------------- #

def classify_vix_regime(vix: float) -> str:
    """Bucket the VIX level into named regimes."""
    if vix < 12:
        return "very_low"
    if vix < 15:
        return "low"
    if vix < 20:
        return "normal"
    if vix < 25:
        return "elevated"
    if vix < 30:
        return "stressed"
    return "high_stress"


def classify_term_structure(vix9d: float, vix: float, vix3m: float) -> dict:
    """
    Classify VIX term structure with explicit relevance to short premium strategies.

    Definitions:
        - contango:        vix9d < vix < vix3m       (calm, theta-friendly)
        - front_inverted:  vix9d > vix, vix < vix3m  (front-end stress only)
        - back_inverted:   vix9d < vix, vix > vix3m  (long end below front — fear at the back)
        - fully_inverted:  vix9d > vix > vix3m       (broad stress)
    """
    front_inverted = vix9d > vix
    back_inverted = vix > vix3m

    if not front_inverted and not back_inverted:
        return {
            "structure": "contango",
            "signal": "calm regime",
            "regime": "favorable for short premium strategies",
        }
    if front_inverted and back_inverted:
        return {
            "structure": "fully_inverted",
            "signal": "broad stress",
            "regime": "unfavorable — short premium has elevated risk",
        }
    if front_inverted:
        return {
            "structure": "front_inverted",
            "signal": "front-end stress, longer-term calm",
            "regime": "caution — short-term vol elevated",
        }
    return {
        "structure": "back_inverted",
        "signal": "long-end stress under a calm front",
        "regime": "watch — fear creeping into longer dates",
    }


# --------------------------------------------------------------------------- #
# Yahoo Finance helpers
# --------------------------------------------------------------------------- #

def _fetch_chart(symbol: str) -> Optional[dict]:
    """Fetch one symbol's chart payload from Yahoo. Returns the result[0] dict or None."""
    url = f"{_YAHOO_BASE}/{symbol}"
    params = {"interval": "1d", "range": "5d"}
    try:
        r = requests.get(url, params=params, headers=_HEADERS, timeout=_TIMEOUT)
        r.raise_for_status()
        payload = r.json()
        if payload.get("chart", {}).get("error"):
            return None
        results = payload.get("chart", {}).get("result")
        if not results:
            return None
        return results[0]
    except (requests.exceptions.RequestException, ValueError):
        return None


def _meta_price(chart: Optional[dict]) -> Optional[float]:
    if not chart:
        return None
    meta = chart.get("meta", {})
    val = meta.get("regularMarketPrice")
    return float(val) if val is not None else None


def _meta_prior_close(chart: Optional[dict]) -> Optional[float]:
    if not chart:
        return None
    meta = chart.get("meta", {})
    val = meta.get("previousClose") or meta.get("chartPreviousClose")
    return float(val) if val is not None else None


def _closes_from_chart(chart: Optional[dict]) -> list[float]:
    if not chart:
        return []
    quote = chart.get("indicators", {}).get("quote", [{}])[0]
    closes = quote.get("close", []) or []
    return [float(c) for c in closes if c is not None]


def _last_close(chart: Optional[dict]) -> Optional[float]:
    closes = _closes_from_chart(chart)
    return closes[-1] if closes else None


# --------------------------------------------------------------------------- #
# CLI for ad-hoc verification
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import json
    snapshot = get_market_regime()
    print(json.dumps(snapshot, indent=2))
