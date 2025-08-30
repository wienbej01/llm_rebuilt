"""
Rolling state containers for PSE-LLM trading system.
Manages bars, swings, MSS, FVGs, TCC/MCS with efficient updates and snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .types import FVG, MCS, MSS, TCC, Bar, SwingPoint


@dataclass
class MarketState:
    """Container for all market state data with rolling windows."""

    # Configuration
    max_bars_1m: int = 1000  # Keep last 1000 1-minute bars
    max_bars_5m: int = 500   # Keep last 500 5-minute bars
    max_swings: int = 50      # Keep last 50 swing points
    max_mss: int = 20         # Keep last 20 market structure shifts
    max_fvgs: int = 30        # Keep last 30 fair value gaps
    max_tcc: int = 10         # Keep last 10 time cycle completions
    max_mcs: int = 5          # Keep last 5 market cycle structures

    # Rolling data containers
    bars_1m: list[Bar] = field(default_factory=list)
    bars_5m: list[Bar] = field(default_factory=list)
    swing_points: list[SwingPoint] = field(default_factory=list)
    mss_list: list[MSS] = field(default_factory=list)
    fvgs: list[FVG] = field(default_factory=list)
    tcc_history: list[TCC] = field(default_factory=list)
    mcs_history: list[MCS] = field(default_factory=list)

    # Current state
    current_tcc: TCC | None = None
    current_mcs: MCS | None = None
    last_update: datetime | None = None

    # Metadata
    symbol: str = ""
    initialized: bool = False
    warmup_complete: bool = False

    def add_1m_bar(self, bar: Bar) -> None:
        """Add a 1-minute bar to the rolling window."""
        self.bars_1m.append(bar)
        if len(self.bars_1m) > self.max_bars_1m:
            self.bars_1m.pop(0)
        self.last_update = bar.timestamp

    def add_5m_bar(self, bar: Bar) -> None:
        """Add a 5-minute bar to the rolling window."""
        self.bars_5m.append(bar)
        if len(self.bars_5m) > self.max_bars_5m:
            self.bars_5m.pop(0)
        self.last_update = bar.timestamp

    def add_swing_point(self, swing: SwingPoint) -> None:
        """Add a swing point to the rolling window."""
        self.swing_points.append(swing)
        if len(self.swing_points) > self.max_swings:
            self.swing_points.pop(0)

    def add_mss(self, mss: MSS) -> None:
        """Add a market structure shift to the rolling window."""
        self.mss_list.append(mss)
        if len(self.mss_list) > self.max_mss:
            self.mss_list.pop(0)

    def add_fvg(self, fvg: FVG) -> None:
        """Add a fair value gap to the rolling window."""
        self.fvgs.append(fvg)
        if len(self.fvgs) > self.max_fvgs:
            self.fvgs.pop(0)

    def add_tcc(self, tcc: TCC) -> None:
        """Add a time cycle completion to the rolling window."""
        self.tcc_history.append(tcc)
        self.current_tcc = tcc
        if len(self.tcc_history) > self.max_tcc:
            self.tcc_history.pop(0)

    def add_mcs(self, mcs: MCS) -> None:
        """Add a market cycle structure to the rolling window."""
        self.mcs_history.append(mcs)
        self.current_mcs = mcs
        if len(self.mcs_history) > self.max_mcs:
            self.mcs_history.pop(0)

    def get_latest_1m_bars(self, n: int) -> list[Bar]:
        """Get the latest n 1-minute bars."""
        return self.bars_1m[-n:] if len(self.bars_1m) >= n else self.bars_1m.copy()

    def get_latest_5m_bars(self, n: int) -> list[Bar]:
        """Get the latest n 5-minute bars."""
        return self.bars_5m[-n:] if len(self.bars_5m) >= n else self.bars_5m.copy()

    def get_recent_swings(self, n: int) -> list[SwingPoint]:
        """Get the most recent n swing points."""
        return self.swing_points[-n:] if len(self.swing_points) >= n else self.swing_points.copy()

    def get_active_fvgs(self) -> list[FVG]:
        """Get all active (unfilled) fair value gaps."""
        return [fvg for fvg in self.fvgs if not fvg.is_filled]

    def get_valid_mss(self) -> list[MSS]:
        """Get all valid market structure shifts."""
        return [mss for mss in self.mss_list if mss.is_valid]

    def clear(self) -> None:
        """Clear all state data."""
        self.bars_1m.clear()
        self.bars_5m.clear()
        self.swing_points.clear()
        self.mss_list.clear()
        self.fvgs.clear()
        self.tcc_history.clear()
        self.mcs_history.clear()
        self.current_tcc = None
        self.current_mcs = None
        self.last_update = None
        self.initialized = False
        self.warmup_complete = False

    def is_warm(self) -> bool:
        """Check if the state has enough data for analysis."""
        min_1m_bars = 50
        min_5m_bars = 20
        min_swings = 5

        return (
            len(self.bars_1m) >= min_1m_bars and
            len(self.bars_5m) >= min_5m_bars and
            len(self.swing_points) >= min_swings
        )

    def snapshot(self) -> dict[str, Any]:
        """Create a snapshot of current state for serialization."""
        return {
            "symbol": self.symbol,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "bars_1m_count": len(self.bars_1m),
            "bars_5m_count": len(self.bars_5m),
            "swing_points_count": len(self.swing_points),
            "mss_count": len(self.mss_list),
            "fvgs_count": len(self.fvgs),
            "tcc_count": len(self.tcc_history),
            "mcs_count": len(self.mcs_history),
            "current_tcc": self.current_tcc.model_dump() if self.current_tcc else None,
            "current_mcs": self.current_mcs.model_dump() if self.current_mcs else None,
            "initialized": self.initialized,
            "warmup_complete": self.warmup_complete,
            "is_warm": self.is_warm()
        }

    def restore_from_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Restore state from a snapshot (for testing/debugging)."""
        self.symbol = snapshot.get("symbol", "")
        self.initialized = snapshot.get("initialized", False)
        self.warmup_complete = snapshot.get("warmup_complete", False)

        if snapshot.get("last_update"):
            self.last_update = datetime.fromisoformat(snapshot["last_update"])


@dataclass
class EngineState:
    """Global engine state container."""

    # Market states by symbol
    market_states: dict[str, MarketState] = field(default_factory=dict)

    # Global counters and metrics
    total_bars_processed: int = 0
    total_setups_generated: int = 0
    total_orders_executed: int = 0
    engine_start_time: datetime | None = None

    # Performance tracking
    avg_processing_time_ms: float = 0.0
    max_processing_time_ms: float = 0.0
    error_count: int = 0

    def get_or_create_market_state(self, symbol: str) -> MarketState:
        """Get or create market state for a symbol."""
        if symbol not in self.market_states:
            self.market_states[symbol] = MarketState(symbol=symbol)
        return self.market_states[symbol]

    def reset(self) -> None:
        """Reset all engine state."""
        self.market_states.clear()
        self.total_bars_processed = 0
        self.total_setups_generated = 0
        self.total_orders_executed = 0
        self.engine_start_time = None
        self.avg_processing_time_ms = 0.0
        self.max_processing_time_ms = 0.0
        self.error_count = 0

    def snapshot(self) -> dict[str, Any]:
        """Create a snapshot of engine state."""
        return {
            "total_bars_processed": self.total_bars_processed,
            "total_setups_generated": self.total_setups_generated,
            "total_orders_executed": self.total_orders_executed,
            "engine_start_time": self.engine_start_time.isoformat() if self.engine_start_time else None,
            "avg_processing_time_ms": self.avg_processing_time_ms,
            "max_processing_time_ms": self.max_processing_time_ms,
            "error_count": self.error_count,
            "symbols": list(self.market_states.keys()),
            "market_snapshots": {
                symbol: state.snapshot()
                for symbol, state in self.market_states.items()
            }
        }


# Global engine state instance
engine_state = EngineState()
