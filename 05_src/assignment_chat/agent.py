"""LLM call layer. Phase 3: pre-filter + chat completion + tool dispatch."""

import os
import json
from openai import OpenAI

from prompts import VEGA_SYSTEM_PROMPT
from guardrails import precheck
from services.quant_utils import (
    price_short_put_spread,
    spread_pnl_at_spot,
    compute_returns_stats,
    compute_trade_stats,
    mae_mfe,
    kelly_position_size,
)
from services.market_regime import get_market_regime
from services.trade_journal import search_trade_journal, get_trade_by_id

client = OpenAI(
    base_url="https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1",
    api_key="any value",
    default_headers={"x-api-key": os.getenv("API_GATEWAY_KEY")},
)
MODEL = "gpt-4o-mini"

_TOOL_REGISTRY = {
    "price_short_put_spread": price_short_put_spread,
    "spread_pnl_at_spot": spread_pnl_at_spot,
    "compute_returns_stats": compute_returns_stats,
    "compute_trade_stats": compute_trade_stats,
    "mae_mfe": mae_mfe,
    "kelly_position_size": kelly_position_size,
    "get_market_regime": get_market_regime,
    "search_trade_journal": search_trade_journal,
    "get_trade_by_id": get_trade_by_id,
}

_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "price_short_put_spread",
            "description": "Price an SPX short put credit spread under Black-Scholes-Merton and return theoretical credit, max loss, breakeven, and net Greeks (delta, gamma, theta/day, vega). Use when the user asks 'what's a [X]/[Y] worth' or wants to evaluate a hypothetical spread.",
            "parameters": {
                "type": "object",
                "properties": {
                    "spot": {"type": "number", "description": "Underlying SPX spot price"},
                    "short_K": {"type": "number", "description": "Short leg strike (sold put, higher strike)"},
                    "long_K": {"type": "number", "description": "Long leg strike (bought put, lower strike). Must be < short_K."},
                    "dte_minutes": {"type": "number", "description": "Minutes to expiration"},
                    "sigma_short": {"type": "number", "description": "Implied vol on short leg as decimal (e.g., 0.15 for 15%)"},
                    "sigma_long": {"type": "number", "description": "Implied vol on long leg as decimal"},
                    "r": {"type": "number", "description": "Risk-free rate as decimal", "default": 0.05},
                    "q": {"type": "number", "description": "Dividend yield as decimal", "default": 0.0},
                },
                "required": ["spot", "short_K", "long_K", "dte_minutes", "sigma_short", "sigma_long"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spread_pnl_at_spot",
            "description": "Compute mark-to-market P&L on an open short put spread at a hypothetical spot level. Use for 'what if SPX drops to X' style questions on a live or recent position.",
            "parameters": {
                "type": "object",
                "properties": {
                    "short_K": {"type": "number"},
                    "long_K": {"type": "number"},
                    "credit_received": {"type": "number", "description": "Credit collected at entry, in dollars per spread (e.g., 1.10)"},
                    "spot_now": {"type": "number", "description": "Hypothetical spot level"},
                    "dte_min_remaining": {"type": "number"},
                    "sigma_short": {"type": "number"},
                    "sigma_long": {"type": "number"},
                    "r": {"type": "number", "default": 0.05},
                    "q": {"type": "number", "default": 0.0},
                },
                "required": ["short_K", "long_K", "credit_received", "spot_now", "dte_min_remaining", "sigma_short", "sigma_long"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_returns_stats",
            "description": "Compute Sharpe, Sortino, vol, drawdown, and other equity-curve statistics on a list of daily P&L values. Use when the user has a series of daily P&Ls and wants performance metrics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "daily_pnl": {"type": "array", "items": {"type": "number"}, "description": "List of daily P&L in dollars, one per trading day"},
                    "capital": {"type": "number", "description": "Optional reference capital for % return calcs"},
                    "periods_per_year": {"type": "integer", "default": 252},
                },
                "required": ["daily_pnl"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_trade_stats",
            "description": "Compute trade-level statistics: hit rate, win/loss ratio, profit factor, expectancy. Use when the user has a list of per-trade P&Ls.",
            "parameters": {
                "type": "object",
                "properties": {
                    "per_trade_pnl": {"type": "array", "items": {"type": "number"}, "description": "List of per-trade P&L values in dollars"},
                },
                "required": ["per_trade_pnl"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mae_mfe",
            "description": "Compute Max Adverse Excursion and Max Favorable Excursion on a single trade's P&L trajectory (minute-by-minute or per-scan). Use when analyzing the path of one trade.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pnl_trajectory": {"type": "array", "items": {"type": "number"}, "description": "Ordered list of P&L during a trade's life, in dollars"},
                },
                "required": ["pnl_trajectory"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_trade_by_id",
            "description": (
                "Fetch a specific trade by its exact ID (format YYYY-MM-DD-HH:MM:SS, e.g., "
                "'2026-05-01-09:58:08'). USE THIS, not search_trade_journal, whenever the user "
                "references a specific trade by ID — including follow-up questions about a trade "
                "just discussed in this conversation. Returns the full markdown doc and metadata "
                "for that exact trade. Returns found=false if the ID is not in the index."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "trade_id": {
                        "type": "string",
                        "description": "Exact trade ID, format YYYY-MM-DD-HH:MM:SS",
                    },
                },
                "required": ["trade_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_trade_journal",
            "description": (
                "Semantic search over the user's SPX trade journal. Use for queries about past trades: "
                "'show me last week's losses', 'trades where the model had high conviction but I lost', "
                "'find similar setups to today's', 'what happened on MONEYNESS_DANGER exits', etc. "
                "Returns the most semantically-similar trades with metadata and the abstracted trade docs. "
                "Filters available: outcome (win/loss/scratch) and exit_reason "
                "(PROFIT_TIER_15, PROFIT_TIER_20, MONEYNESS_DANGER, VELOCITY_EXIT)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural-language search query"},
                    "k": {"type": "integer", "default": 4, "description": "Number of trades to return (1-10)"},
                    "outcome": {
                        "type": "string",
                        "enum": ["win", "loss", "scratch"],
                        "description": "Optional outcome filter",
                    },
                    "exit_reason": {
                        "type": "string",
                        "enum": ["PROFIT_TIER_15", "PROFIT_TIER_20", "MONEYNESS_DANGER", "VELOCITY_EXIT"],
                        "description": "Optional exit reason filter",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_regime",
            "description": (
                "Fetch a live market regime snapshot from Yahoo Finance: SPX level + day change, "
                "VIX level + regime classification, VIX term structure (VIX9D/VIX/VIX3M with explicit "
                "contango/backwardation signal for short premium strategies), and 10Y Treasury yield. "
                "Use whenever the user asks about the current market state, regime, what's the tape doing, "
                "VIX, term structure, macro context, or 'should I be trading right now'. "
                "Returns synthesized data — narrate it in voice, lead with the term structure signal "
                "when discussing short premium suitability. If a field is None, the data is unavailable; "
                "say so plainly rather than guessing."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "kelly_position_size",
            "description": "Compute Kelly-based position sizing. Returns recommended contracts at full Kelly and at a fractional Kelly multiplier (default quarter-Kelly). Use for sizing discussions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "win_rate": {"type": "number", "description": "Probability of winning trade (0-1)"},
                    "avg_win": {"type": "number", "description": "Average win in dollars per contract"},
                    "avg_loss": {"type": "number", "description": "Average loss in dollars per contract (positive number)"},
                    "account_equity": {"type": "number"},
                    "max_loss_per_contract": {"type": "number", "description": "Max loss per spread in dollars (e.g., 5pt width = $400 max loss after $100 credit)"},
                    "kelly_multiplier": {"type": "number", "default": 0.25, "description": "Fraction of full Kelly to use (default 0.25)"},
                },
                "required": ["win_rate", "avg_win", "avg_loss", "account_equity", "max_loss_per_contract"],
            },
        },
    },
]

MAX_TOOL_ITERATIONS = 5


def chat(history: list[dict], user_message: str) -> str:
    """Send a turn to Vega. Pre-filter → LLM → optional tool dispatch loop → final reply."""
    refusal = precheck(user_message)
    if refusal is not None:
        return refusal

    messages = [{"role": "system", "content": VEGA_SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    for _ in range(MAX_TOOL_ITERATIONS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=_TOOLS_SCHEMA,
            temperature=0.3,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            return msg.content or ""

        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ],
        })

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
                fn = _TOOL_REGISTRY[fn_name]
                result = fn(**args)
                content = json.dumps(result)
            except Exception as e:
                content = json.dumps({"error": str(e), "tool": fn_name})

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": content,
            })

    return "Tool-call loop exceeded iteration cap. Try rephrasing the question."
