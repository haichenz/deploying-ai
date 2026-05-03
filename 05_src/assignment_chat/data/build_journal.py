"""Trade journal builder.

Parses SPX trade log files into structured trade records and embeds them
into a persistent ChromaDB collection for semantic search by the chat agent.

Two abstraction modes are supported. The public mode buckets numeric features
into categorical labels and replaces minute-by-minute trajectories with shape
descriptors. The alternate mode passes structured data through with full
fidelity. Mode is selected via the --mode CLI flag.

Usage:
    python data/build_journal.py --mode public --input data/sample_logs/
"""

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import chromadb


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parent  # data/

PATHS = {
    "public": {
        "docs": ROOT / "trade_docs",
        "chroma": ROOT / "chroma_db",
        "collection": "trade_journal_public",
    },
    "private": {
        "docs": ROOT / "trade_docs_private",
        "chroma": ROOT / "chroma_db_private",
        "collection": "trade_journal_private",
    },
}


# --------------------------------------------------------------------------- #
# Section 1 — Parser (mode-agnostic)
# --------------------------------------------------------------------------- #

@dataclass
class TrajectoryPoint:
    minute: int
    mid: float
    otm: int
    pnl: float


@dataclass
class Trade:
    """One closed trade reconstructed from log events."""
    trade_id: str                       # YYYY-MM-DD-HH:MM:SS 
    trade_date: str                     # YYYY-MM-DD
    session_vix: Optional[float] = None
    entry_time: Optional[str] = None
    entry_short_K: Optional[int] = None
    entry_long_K: Optional[int] = None
    entry_credit: Optional[float] = None
    entry_short_delta: Optional[float] = None  # from PRE_FILL_CHECK |delta|
    entry_otm_pts: Optional[float] = None
    entry_iv_rank: Optional[float] = None
    entry_ensemble: Optional[float] = None
    exit_time: Optional[str] = None
    exit_reason: Optional[str] = None
    exit_debit: Optional[float] = None
    exit_pnl: Optional[float] = None
    hold_minutes: Optional[int] = None
    trajectory: list[TrajectoryPoint] = field(default_factory=list)
    outcome: Optional[str] = None       # "win" | "loss" | "scratch"


_RE_SESSION_VIX = re.compile(r"SESSION START.*?VIX=([\d.]+)")
_RE_PRE_FILL = re.compile(
    r"^(\d{2}:\d{2}:\d{2})\s+PRE_FILL_CHECK\s+\|\s+(\d+)/(\d+)\s+\|\s+PASS\s+\|\s+\|delta\|=([\d.]+)\s+\|\s+otm=([\d.]+)pts"
)
_RE_FILLED = re.compile(
    r"^(\d{2}:\d{2}:\d{2})\s+FILLED\s+\|\s+(\d+)/(\d+)\s+\|\s+credit=\$([\d.]+)"
)
_RE_ENTRY = re.compile(
    r"^(\d{2}:\d{2}:\d{2})\s+ENTRY\s+\|\s+(\d+)/(\d+)\s+\|\s+credit=\$([\d.]+)\s+\|\s+ens=([\d.]+)"
)
_RE_EVAL = re.compile(
    r"^(\d{2}:\d{2}:\d{2})\s+EVAL\s+\|\s+(\d+)/(\d+)\s+mid=\$[\d.]+\s+iv_rank=([\d.]+)\s+ens=([\d.]+)\s+→\s+ENTER"
)
_RE_POS = re.compile(
    r"^(\d{2}:\d{2}:\d{2})\s+POS\s+\|\s+(\d+)/(\d+)\s+mid=\$([\d.]+)\s+OTM=(-?\d+)\s+PnL=\$([+-]?\d+(?:\.\d+)?)\s+held=(\d+)min"
)
_RE_EXIT = re.compile(
    r"^(\d{2}:\d{2}:\d{2})\s+EXIT\s+\|\s+(\d+)/(\d+)\s+\|\s+(\w+)"
)
_RE_EXIT_FILLED = re.compile(
    r"^(\d{2}:\d{2}:\d{2})\s+EXIT FILLED.*?\|\s+(\d+)/(\d+)\s+\|\s+debit=\$([\d.]+)\s+\|\s+PnL=\$([+-]?[\d.]+)"
)


def parse_log(log_path: Path) -> list[Trade]:
    """
    Parse one daily .log file into a list of closed Trade objects.

    Open-position state machine:


    Multiple trades can be open if the system re-enters quickly. Track by strike pair.
    """
    trade_date = _extract_date_from_filename(log_path)
    text = log_path.read_text()

    session_vix = None
    m = _RE_SESSION_VIX.search(text)
    if m:
        session_vix = float(m.group(1))

    open_trades: dict[tuple[int, int], Trade] = {}
    closed_trades: list[Trade] = []
    pending_pre_fill: dict[tuple[int, int], dict] = {}
    pending_eval_iv_rank: dict[tuple[int, int], float] = {}

    for line in text.splitlines():
        m = _RE_PRE_FILL.match(line)
        if m:
            short_K = int(m.group(2))
            long_K = int(m.group(3))
            pending_pre_fill[(short_K, long_K)] = {
                "delta": float(m.group(4)),
                "otm_pts": float(m.group(5)),
            }
            continue

        m = _RE_EVAL.match(line)
        if m:
            short_K = int(m.group(2))
            long_K = int(m.group(3))
            iv_rank = float(m.group(4))
            # EVAL may come before or after ENTRY for the same strike pair.
            # If the trade is already open, update it directly; otherwise buffer.
            if (short_K, long_K) in open_trades:
                open_trades[(short_K, long_K)].entry_iv_rank = iv_rank
            else:
                pending_eval_iv_rank[(short_K, long_K)] = iv_rank
            continue

        m = _RE_ENTRY.match(line)
        if m:
            time_s = m.group(1)
            short_K = int(m.group(2))
            long_K = int(m.group(3))
            credit = float(m.group(4))
            ensemble = float(m.group(5))

            trade = Trade(
                trade_id=f"{trade_date}-{time_s}",
                trade_date=trade_date,
                session_vix=session_vix,
                entry_time=time_s,
                entry_short_K=short_K,
                entry_long_K=long_K,
                entry_credit=credit,
                entry_ensemble=ensemble,
            )
            pf = pending_pre_fill.pop((short_K, long_K), None)
            if pf:
                trade.entry_short_delta = pf["delta"]
                trade.entry_otm_pts = pf["otm_pts"]
            iv = pending_eval_iv_rank.pop((short_K, long_K), None)
            if iv is not None:
                trade.entry_iv_rank = iv

            open_trades[(short_K, long_K)] = trade
            continue

        m = _RE_POS.match(line)
        if m:
            short_K = int(m.group(2))
            long_K = int(m.group(3))
            trade = open_trades.get((short_K, long_K))
            if trade is None:
                continue
            trade.trajectory.append(TrajectoryPoint(
                minute=int(m.group(7)),
                mid=float(m.group(4)),
                otm=int(m.group(5)),
                pnl=float(m.group(6)),
            ))
            continue

        m = _RE_EXIT.match(line)
        if m:
            short_K = int(m.group(2))
            long_K = int(m.group(3))
            reason = m.group(4)
            trade = open_trades.get((short_K, long_K))
            if trade is not None:
                trade.exit_reason = reason
            continue

        m = _RE_EXIT_FILLED.match(line)
        if m:
            time_s = m.group(1)
            short_K = int(m.group(2))
            long_K = int(m.group(3))
            debit = float(m.group(4))
            pnl = float(m.group(5))

            trade = open_trades.pop((short_K, long_K), None)
            if trade is None:
                continue

            trade.exit_time = time_s
            trade.exit_debit = debit
            trade.exit_pnl = pnl
            trade.hold_minutes = trade.trajectory[-1].minute if trade.trajectory else None

            if pnl > 0.5:
                trade.outcome = "win"
            elif pnl < -0.5:
                trade.outcome = "loss"
            else:
                trade.outcome = "scratch"

            closed_trades.append(trade)
            continue

    return closed_trades


def _extract_date_from_filename(p: Path) -> str:
    """Expect filenames like spx_2026-05-01.log. Fall back to file mtime if not parseable."""
    m = re.search(r"(\d{4}-\d{2}-\d{2})", p.name)
    if m:
        return m.group(1)
    from datetime import datetime
    return datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d")


# --------------------------------------------------------------------------- #
# Section 2 — Abstractor (mode-divergent)
# --------------------------------------------------------------------------- #

PUBLIC_BUCKETS = {
    "delta": [
        (0.30, "high_delta"),
        (0.20, "mid_delta"),
        (0.0,  "low_delta"),
    ],
    "ensemble": [
        (0.65, "strong_signal"),
        (0.50, "neutral_signal"),
        (0.0,  "weak_signal"),
    ],
    "iv_rank": [
        (0.30, "high_iv"),
        (0.10, "mid_iv"),
        (0.0,  "low_iv"),
    ],
    "vix": [
        (22.0, "elevated_vix"),
        (15.0, "normal_vix"),
        (0.0,  "low_vix"),
    ],
    "pnl": [
        (50.0,  "win_large"),
        (0.5,   "win_small"),
        (-0.5,  "scratch"),
        (-50.0, "loss_small"),
        (-1e9,  "loss_large"),
    ],
    "hold": [
        (20, "slow"),
        (5,  "medium"),
        (0,  "fast"),
    ],
    "otm": [
        (15.0, "wide_otm"),
        (8.0,  "moderate_otm"),
        (0.0,  "tight_otm"),
    ],
}


def _bucket(value: Optional[float], scale: str, take_abs: bool = False) -> str:
    if value is None:
        return "unknown"
    v = abs(value) if take_abs else value
    for threshold, label in PUBLIC_BUCKETS[scale]:
        if v >= threshold:
            return label
    return "unknown"


def _trajectory_shape(trajectory: list[TrajectoryPoint]) -> str:
    """Categorize a P&L trajectory into a shape descriptor."""
    if not trajectory:
        return "no_trajectory"

    pnls = [p.pnl for p in trajectory]
    final = pnls[-1]
    peak = max(pnls)
    trough = min(pnls)

    if final > 0 and trough >= -5:
        return "clean_win"
    if final > 0 and trough < -5:
        return "drawdown_then_recovery"
    if final < 0 and peak > 5:
        return "gave_back_gains"
    if final < 0 and peak <= 5:
        return "straight_loss"
    return "choppy"


def abstract_trade_public(t: Trade) -> dict:
    """Public mode: bucketed numerics, shape descriptor, strikes/dates kept."""
    return {
        "trade_id": t.trade_id,
        "trade_date": t.trade_date,
        "outcome": t.outcome or "unknown",
        "exit_reason": t.exit_reason or "unknown",
        "session_regime": _bucket(t.session_vix, "vix"),
        "entry_delta_band": _bucket(t.entry_short_delta, "delta", take_abs=True),
        "entry_otm_band": _bucket(t.entry_otm_pts, "otm"),
        "entry_iv_band": _bucket(t.entry_iv_rank, "iv_rank"),
        "entry_conviction": _bucket(t.entry_ensemble, "ensemble"),
        "pnl_band": _bucket(t.exit_pnl, "pnl"),
        "hold_band": _bucket(t.hold_minutes, "hold"),
        "trajectory_shape": _trajectory_shape(t.trajectory),
        "short_K": t.entry_short_K,
        "long_K": t.entry_long_K,
    }


def abstract_trade_private(t: Trade) -> dict:
    """Alternate mode: full-fidelity passthrough."""
    return asdict(t)


# --------------------------------------------------------------------------- #
# Section 2b — Markdown rendering
# --------------------------------------------------------------------------- #

def render_doc_public(trade: Trade, abstracted: dict) -> str:
    """Markdown doc for public mode. Skips bullets and narrative clauses
    whose abstracted value is 'unknown' to avoid surfacing parser gaps in
    downstream chat output.
    """
    field_map = [
        ("entry_delta_band", "Short leg delta band"),
        ("entry_otm_band", "OTM distance band"),
        ("entry_iv_band", "IV environment"),
        ("entry_conviction", "Model conviction"),
    ]
    bullets = [
        f"- {label}: {abstracted[key]}"
        for key, label in field_map
        if abstracted.get(key) and abstracted[key] != "unknown"
    ]
    if not bullets:
        bullets = ["- (entry signal context not captured)"]
    entry_section = "\n".join(bullets)

    conviction = abstracted.get("entry_conviction", "unknown")
    regime = abstracted.get("session_regime", "unknown")
    iv_band = abstracted.get("entry_iv_band", "unknown")
    delta = abstracted.get("entry_delta_band", "unknown")
    otm = abstracted.get("entry_otm_band", "unknown")

    entry_clauses = []
    if conviction != "unknown":
        entry_clauses.append(f"{conviction} model conviction")
    if regime != "unknown":
        entry_clauses.append(f"in a {regime} session")
    if iv_band != "unknown":
        entry_clauses.append(f"under {iv_band} IV")

    if entry_clauses:
        entry_sentence = "Entered with " + ", ".join(entry_clauses) + "."
    else:
        entry_sentence = "Entry signal context was not captured."

    leg_clauses = []
    if delta != "unknown":
        leg_clauses.append(f"{delta} band")
    if otm != "unknown":
        leg_clauses.append(f"{otm} OTM at fill")
    leg_sentence = (
        f"Short leg in the {', '.join(leg_clauses)}." if leg_clauses else ""
    )

    return f"""# Trade {abstracted['trade_date']} — {abstracted['short_K']}/{abstracted['long_K']}

**Outcome:** {abstracted['outcome']} ({abstracted['pnl_band']})
**Exit reason:** {abstracted['exit_reason']}
**Session regime:** {abstracted['session_regime']}

## Entry context
{entry_section}

## Lifecycle
- Hold time: {abstracted['hold_band']}
- Trajectory shape: {abstracted['trajectory_shape']}

## Narrative
Intraday SPX put credit spread. {entry_sentence} {leg_sentence} Held for a {abstracted['hold_band']} duration; exit triggered by {abstracted['exit_reason']}. Trajectory was {abstracted['trajectory_shape']}, landing in the {abstracted['pnl_band']} P&L band.
"""


def render_doc_private(trade: Trade, _abstracted: dict) -> str:
    """Markdown doc for the alternate mode. Full numeric fidelity."""
    traj_lines = "\n".join(
        f"- min {p.minute}: mid=${p.mid:.2f} OTM={p.otm} P&L=${p.pnl:+.0f}"
        for p in trade.trajectory
    )
    return f"""# Trade {trade.trade_id}

**Outcome:** {trade.outcome}  |  P&L: ${trade.exit_pnl:+.2f}  |  Exit: {trade.exit_reason}
**Date:** {trade.trade_date}  |  Session VIX: {trade.session_vix}

## Entry
- Time: {trade.entry_time}
- Spread: {trade.entry_short_K}/{trade.entry_long_K}
- Credit: ${trade.entry_credit:.2f}
- Short delta: {trade.entry_short_delta}
- OTM at entry: {trade.entry_otm_pts} pts
- IV rank: {trade.entry_iv_rank}
- Ensemble score: {trade.entry_ensemble}

## Exit
- Time: {trade.exit_time}
- Hold: {trade.hold_minutes} minutes
- Exit debit: ${trade.exit_debit:.2f}
- Reason: {trade.exit_reason}

## Trajectory
{traj_lines}
"""


# --------------------------------------------------------------------------- #
# Section 3 — Embedder + persistence
# --------------------------------------------------------------------------- #

def build_collection(mode: str, trades: list[Trade], reset: bool = True) -> int:
    """
    Embed trades into a persistent Chroma collection for the given mode.
    Returns the count of documents added.
    """
    cfg = PATHS[mode]
    docs_dir: Path = cfg["docs"]
    chroma_dir: Path = cfg["chroma"]
    coll_name: str = cfg["collection"]

    if reset:
        if docs_dir.exists():
            shutil.rmtree(docs_dir)
        if chroma_dir.exists():
            shutil.rmtree(chroma_dir)

    docs_dir.mkdir(parents=True, exist_ok=True)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    abstractor = abstract_trade_public if mode == "public" else abstract_trade_private
    renderer = render_doc_public if mode == "public" else render_doc_private

    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    for t in trades:
        abstracted = abstractor(t)
        md = renderer(t, abstracted)

        out_path = docs_dir / f"{t.trade_id}.md"
        out_path.write_text(md)

        documents.append(md)
        metadatas.append({
            "trade_date": t.trade_date,
            "outcome": t.outcome or "unknown",
            "exit_reason": t.exit_reason or "unknown",
            "short_K": int(t.entry_short_K) if t.entry_short_K is not None else 0,
            "long_K": int(t.entry_long_K) if t.entry_long_K is not None else 0,
            "session_vix_band": _bucket(t.session_vix, "vix"),
        })
        ids.append(t.trade_id)

    client = chromadb.PersistentClient(path=str(chroma_dir))
    try:
        client.delete_collection(name=coll_name)
    except Exception:
        pass
    collection = client.create_collection(name=coll_name)

    if documents:
        collection.add(documents=documents, metadatas=metadatas, ids=ids)

    return len(documents)


# --------------------------------------------------------------------------- #
# Section 4 — CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Build the SPX trade journal.")
    parser.add_argument("--mode", choices=["public", "private"], default="public")
    parser.add_argument("--input", type=Path, required=True,
                        help="Directory containing .log files")
    parser.add_argument("--no-reset", action="store_true",
                        help="Append to existing collection instead of rebuilding")
    args = parser.parse_args()

    if not args.input.is_dir():
        raise SystemExit(f"Input directory not found: {args.input}")

    log_files = sorted(args.input.glob("*.log"))
    if not log_files:
        raise SystemExit(f"No .log files found in {args.input}")

    print(f"Mode: {args.mode}")
    print(f"Found {len(log_files)} log file(s)")

    all_trades: list[Trade] = []
    for log_path in log_files:
        trades = parse_log(log_path)
        print(f"  {log_path.name}: {len(trades)} closed trades")
        all_trades.extend(trades)

    if not all_trades:
        raise SystemExit("No trades parsed. Check log format.")

    count = build_collection(args.mode, all_trades, reset=not args.no_reset)
    print(f"Indexed {count} trades into {PATHS[args.mode]['collection']}")
    print(f"Markdown docs: {PATHS[args.mode]['docs']}")
    print(f"Chroma DB:     {PATHS[args.mode]['chroma']}")


if __name__ == "__main__":
    main()
