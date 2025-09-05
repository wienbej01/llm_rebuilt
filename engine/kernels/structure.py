"""
Structure kernel for PSE-LLM trading system.
Detects and updates market structure elements: swings, MSS, FVGs.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numba import float64, int64, jit

from engine.state import MarketState
from engine.trading_types import FVG, MSS, Bar, FVGType, MSSDirection, SwingPoint, SwingType

logger = logging.getLogger(__name__)


class StructureKernel:
    """Kernel for detecting and managing market structure elements."""

    def __init__(self, swing_lookback: int = 20, fvg_threshold: float = 0.001):
        """
        Initialize structure kernel.

        Args:
            swing_lookback: Number of bars to look back for swing detection
            fvg_threshold: Minimum gap size for FVG detection (as fraction of price)
        """
        self.swing_lookback = swing_lookback
        self.fvg_threshold = fvg_threshold

    def update_structure(self, market_state: MarketState, new_bars: list[Bar]) -> None:
        """
        Update market structure with new bars.

        Args:
            market_state: Current market state
            new_bars: New bars to process
        """
        # Detect swings
        self._detect_swings(market_state)

        # Detect MSS
        self._detect_mss(market_state)

        # Detect FVGs
        self._detect_fvgs(market_state)

        # Update FVG states
        self._update_fvg_states(market_state)

        logger.debug(f"Updated structure: {len(market_state.swing_points)} swings, {len(market_state.mss_list)} MSS, {len(market_state.fvgs)} FVGs")

    def _detect_swings(self, market_state: MarketState) -> None:
        """Detect swing points in the market structure."""
        if len(market_state.bars_5m) < self.swing_lookback:
            return

        # Get recent bars for swing detection
        recent_bars = market_state.get_latest_5m_bars(self.swing_lookback)

        # Find potential swing highs and lows
        for i in range(2, len(recent_bars) - 2):
            current_bar = recent_bars[i]
            current_price = current_bar.close

            # Check for swing high
            if self._is_swing_high(recent_bars, i):
                swing = SwingPoint(
                    bar_index=len(market_state.bars_5m) - len(recent_bars) + i,
                    timestamp=current_bar.timestamp,
                    price=current_bar.high,
                    swing_type=SwingType.SWING_HIGH,
                    strength=self._calculate_swing_strength(recent_bars, i, SwingType.SWING_HIGH)
                )
                market_state.add_swing_point(swing)

            # Check for swing low
            if self._is_swing_low(recent_bars, i):
                swing = SwingPoint(
                    bar_index=len(market_state.bars_5m) - len(recent_bars) + i,
                    timestamp=current_bar.timestamp,
                    price=current_bar.low,
                    swing_type=SwingType.SWING_LOW,
                    strength=self._calculate_swing_strength(recent_bars, i, SwingType.SWING_LOW)
                )
                market_state.add_swing_point(swing)

    def _is_swing_high(self, bars: list[Bar], index: int) -> bool:
        """Check if bar at index is a swing high."""
        if index < 2 or index >= len(bars) - 2:
            return False

        current_high = bars[index].high
        left_higher = all(bars[i].high < current_high for i in range(index - 2, index))
        right_higher = all(bars[i].high < current_high for i in range(index + 1, index + 3))

        return left_higher and right_higher

    def _is_swing_low(self, bars: list[Bar], index: int) -> bool:
        """Check if bar at index is a swing low."""
        if index < 2 or index >= len(bars) - 2:
            return False

        current_low = bars[index].low
        left_lower = all(bars[i].low > current_low for i in range(index - 2, index))
        right_lower = all(bars[i].low > current_low for i in range(index + 1, index + 3))

        return left_lower and right_lower

    def _calculate_swing_strength(self, bars: list[Bar], index: int, swing_type: SwingType) -> int:
        """Calculate swing point strength (1-10)."""
        if swing_type == SwingType.SWING_HIGH:
            current_price = bars[index].high
            # Look at how many bars it dominates
            left_count = sum(1 for i in range(index - 1, max(-1, index - 10), -1)
                           if i >= 0 and bars[i].high < current_price)
            right_count = sum(1 for i in range(index + 1, min(len(bars), index + 10))
                            if bars[i].high < current_price)
        else:  # SWING_LOW
            current_price = bars[index].low
            left_count = sum(1 for i in range(index - 1, max(-1, index - 10), -1)
                           if i >= 0 and bars[i].low > current_price)
            right_count = sum(1 for i in range(index + 1, min(len(bars), index + 10))
                            if bars[i].low > current_price)

        total_count = left_count + right_count
        return min(10, max(1, total_count // 2))

    def _detect_mss(self, market_state: MarketState) -> None:
        """Detect Market Structure Shifts (MSS)."""
        if len(market_state.swing_points) < 2:
            return

        # Get recent swing points
        recent_swings = market_state.get_recent_swings(10)

        # Look for trend changes
        for i in range(1, len(recent_swings)):
            current_swing = recent_swings[i]
            previous_swing = recent_swings[i - 1]

            # Check if swing types are different (high->low or low->high)
            if current_swing.swing_type != previous_swing.swing_type:
                # Determine MSS direction
                if (current_swing.swing_type == SwingType.SWING_LOW and
                    previous_swing.swing_type == SwingType.SWING_HIGH):
                    direction = MSSDirection.BULLISH
                    break_price = current_swing.price
                    confirmation_price = previous_swing.price
                elif (current_swing.swing_type == SwingType.SWING_HIGH and
                      previous_swing.swing_type == SwingType.SWING_LOW):
                    direction = MSSDirection.BEARISH
                    break_price = current_swing.price
                    confirmation_price = previous_swing.price
                else:
                    continue

                # Find bar indices
                start_bar = min(current_swing.bar_index, previous_swing.bar_index)
                end_bar = max(current_swing.bar_index, previous_swing.bar_index)

                # Create MSS
                mss = MSS(
                    start_bar=start_bar,
                    end_bar=end_bar,
                    direction=direction,
                    break_price=break_price,
                    confirmation_price=confirmation_price,
                    is_valid=True
                )

                market_state.add_mss(mss)

    def _detect_fvgs(self, market_state: MarketState) -> None:
        """Detect Fair Value Gaps (FVGs)."""
        if len(market_state.bars_5m) < 3:
            return

        recent_bars = market_state.get_latest_5m_bars(20)

        for i in range(2, len(recent_bars)):
            bar1, bar2, bar3 = recent_bars[i-2], recent_bars[i-1], recent_bars[i]

            # Check for bullish FVG: bar1_high < bar3_low
            if bar1.high < bar3.low:
                gap_size = (bar3.low - bar1.high) / bar1.high
                if gap_size > self.fvg_threshold:
                    fvg = FVG(
                        start_bar=len(market_state.bars_5m) - len(recent_bars) + i - 2,
                        end_bar=len(market_state.bars_5m) - len(recent_bars) + i,
                        fvg_type=FVGType.BULLISH,
                        top=bar3.low,
                        bottom=bar1.high,
                        is_filled=False
                    )
                    market_state.add_fvg(fvg)

            # Check for bearish FVG: bar1_low > bar3_high
            elif bar1.low > bar3.high:
                gap_size = (bar1.low - bar3.high) / bar1.low
                if gap_size > self.fvg_threshold:
                    fvg = FVG(
                        start_bar=len(market_state.bars_5m) - len(recent_bars) + i - 2,
                        end_bar=len(market_state.bars_5m) - len(recent_bars) + i,
                        fvg_type=FVGType.BEARISH,
                        top=bar1.low,
                        bottom=bar3.high,
                        is_filled=False
                    )
                    market_state.add_fvg(fvg)

    def _update_fvg_states(self, market_state: MarketState) -> None:
        """Update FVG states (filled, staleness)."""
        if not market_state.fvgs or not market_state.bars_5m:
            return

        latest_bar = market_state.bars_5m[-1]

        for fvg in market_state.fvgs:
            if fvg.is_filled:
                continue

            # Check if FVG is filled
            if fvg.fvg_type == FVGType.BULLISH:
                if latest_bar.high >= fvg.top:
                    fvg.is_filled = True
                    fvg.fill_bar = len(market_state.bars_5m) - 1
            else:  # BEARISH
                if latest_bar.low <= fvg.bottom:
                    fvg.is_filled = True
                    fvg.fill_bar = len(market_state.bars_5m) - 1

    def get_fvg_staleness(self, market_state: MarketState, fvg: FVG) -> float:
        """Calculate FVG staleness (0-1, where 1 is most stale)."""
        if not market_state.bars_5m:
            return 1.0

        current_bar = len(market_state.bars_5m) - 1
        bars_since_creation = current_bar - fvg.end_bar

        # Staleness increases with time, max at 100 bars
        return min(1.0, bars_since_creation / 100.0)

    def get_fvg_retest_info(self, market_state: MarketState, fvg: FVG) -> dict[str, Any]:
        """Get FVG retest information."""
        if not market_state.bars_5m:
            return {"has_retest": False, "retest_price": None, "retest_bar": None}

        # Look for price action that retests the FVG
        for i in range(fvg.start_bar, len(market_state.bars_5m)):
            bar = market_state.bars_5m[i]

            if fvg.fvg_type == FVGType.BULLISH:
                # Bullish FVG retest: price touches bottom of FVG
                if bar.low <= fvg.bottom <= bar.high:
                    return {
                        "has_retest": True,
                        "retest_price": fvg.bottom,
                        "retest_bar": i
                    }
            else:  # BEARISH
                # Bearish FVG retest: price touches top of FVG
                if bar.low <= fvg.top <= bar.high:
                    return {
                        "has_retest": True,
                        "retest_price": fvg.top,
                        "retest_bar": i
                    }

        return {"has_retest": False, "retest_price": None, "retest_bar": None}

    def get_structure_summary(self, market_state: MarketState) -> dict[str, Any]:
        """Get summary of current market structure."""
        return {
            "swing_count": len(market_state.swing_points),
            "mss_count": len(market_state.mss_list),
            "fvg_count": len(market_state.fvgs),
            "active_fvgs": len(market_state.get_active_fvgs()),
            "valid_mss": len(market_state.get_valid_mss()),
            "latest_swing": market_state.swing_points[-1] if market_state.swing_points else None,
            "latest_mss": market_state.mss_list[-1] if market_state.mss_list else None,
            "current_trend": self._determine_current_trend(market_state)
        }

    def _determine_current_trend(self, market_state: MarketState) -> str:
        """Determine current trend direction."""
        valid_mss = market_state.get_valid_mss()
        if not valid_mss:
            return "unknown"

        latest_mss = valid_mss[-1]
        return latest_mss.direction.value.lower()


# Numba-optimized functions for performance
@jit(float64[:](float64[:], int64), nopython=True)
def calculate_swing_strength_numba(prices: np.ndarray, window: int) -> np.ndarray:
    """Numba-optimized swing strength calculation."""
    n = len(prices)
    strength = np.zeros(n)

    for i in range(window, n - window):
        left_count = 0
        right_count = 0

        # Count higher highs/lows to the left
        for j in range(1, window + 1):
            if prices[i - j] < prices[i]:
                left_count += 1

        # Count higher highs/lows to the right
        for j in range(1, window + 1):
            if i + j < n and prices[i + j] < prices[i]:
                right_count += 1

        strength[i] = (left_count + right_count) / (2.0 * window)

    return strength


@jit(float64[:](float64[:], float64[:], float64[:], float64), nopython=True)
def detect_fvgs_numba(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, threshold: float) -> np.ndarray:
    """Numba-optimized FVG detection."""
    n = len(highs)
    fvgs = np.zeros(n)

    for i in range(2, n):
        # Bullish FVG
        if highs[i-2] < lows[i]:
            gap_size = (lows[i] - highs[i-2]) / highs[i-2]
            if gap_size > threshold:
                fvgs[i] = 1.0  # Bullish FVG

        # Bearish FVG
        elif lows[i-2] > highs[i]:
            gap_size = (lows[i-2] - highs[i]) / lows[i-2]
            if gap_size > threshold:
                fvgs[i] = -1.0  # Bearish FVG

    return fvgs
