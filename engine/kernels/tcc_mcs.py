"""
TCC/MCS kernel for PSE-LLM trading system.
Implements Time Cycle Completion (TCC) and Market Cycle Structure (MCS) analysis.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import logging

import numpy as np
from numba import jit, float64, int64

from engine.types import Bar, TCC, MCS
from engine.state import MarketState

logger = logging.getLogger(__name__)


class TrendState(str, Enum):
    """Trend state enumeration."""
    TREND_UP = "TrendUp"
    TREND_DOWN = "TrendDown"
    RANGE = "Range"
    UNKNOWN = "Unknown"


class CyclePhase(str, Enum):
    """Market cycle phase enumeration."""
    ACCUMULATION = "Accumulation"
    MARKUP = "Markup"
    DISTRIBUTION = "Distribution"
    MARKDOWN = "Markdown"
    UNKNOWN = "Unknown"


class TCCMCSKernel:
    """Kernel for TCC and MCS analysis."""

    def __init__(
        self,
        cycle_lookback: int = 50,
        trend_lookback: int = 20,
        volatility_window: int = 14
    ):
        """
        Initialize TCC/MCS kernel.

        Args:
            cycle_lookback: Number of bars to look back for cycle analysis
            trend_lookback: Number of bars to look back for trend analysis
            volatility_window: Window for volatility calculation
        """
        self.cycle_lookback = cycle_lookback
        self.trend_lookback = trend_lookback
        self.volatility_window = volatility_window

    def update_tcc_mcs(self, market_state: MarketState) -> None:
        """
        Update TCC and MCS analysis for the market state.

        Args:
            market_state: Current market state to update
        """
        if len(market_state.bars_5m) < max(self.cycle_lookback, self.trend_lookback):
            return

        # Update TCC analysis
        self._update_tcc(market_state)

        # Update MCS analysis
        self._update_mcs(market_state)

        logger.debug(f"Updated TCC/MCS: {market_state.current_tcc}, {market_state.current_mcs}")

    def _update_tcc(self, market_state: MarketState) -> None:
        """Update Time Cycle Completion analysis."""
        recent_bars = market_state.get_latest_5m_bars(self.cycle_lookback)

        # Detect potential cycle completions
        cycle_completions = self._detect_cycle_completions(recent_bars)

        if cycle_completions:
            # Use the most recent cycle completion
            latest_cycle = cycle_completions[-1]
            market_state.current_tcc = latest_cycle
            market_state.add_tcc(latest_cycle)

    def _update_mcs(self, market_state: MarketState) -> None:
        """Update Market Cycle Structure analysis."""
        recent_bars = market_state.get_latest_5m_bars(self.trend_lookback)

        # Analyze current market structure
        trend_state = self._analyze_trend_state(recent_bars)
        cycle_phase = self._determine_cycle_phase(market_state, recent_bars)
        key_levels = self._identify_key_levels(market_state, recent_bars)

        # Create MCS object
        mcs = MCS(
            timeframe="5m",
            structure_type=trend_state.value,
            bias=cycle_phase.value,
            key_levels=key_levels,
            last_updated=datetime.now(timezone.utc)
        )

        market_state.current_mcs = mcs
        market_state.add_mcs(mcs)

    def _detect_cycle_completions(self, bars: List[Bar]) -> List[TCC]:
        """Detect Time Cycle Completions in the bar series."""
        if len(bars) < 10:
            return []

        completions = []

        # Look for cycle patterns (simplified implementation)
        # In practice, this would involve more sophisticated cycle detection
        closes = [bar.close for bar in bars]

        # Find local extrema that could indicate cycle completion
        for i in range(5, len(closes) - 5):
            # Check for local maximum (potential cycle peak)
            if (closes[i] > closes[i-1] and closes[i] > closes[i+1] and
                closes[i] > closes[i-2] and closes[i] > closes[i+2]):

                # Calculate cycle characteristics
                cycle_length = self._estimate_cycle_length(bars, i)
                cycle_type = self._classify_cycle_type(bars, i)
                strength = self._calculate_cycle_strength(bars, i)

                if cycle_length > 0 and strength > 3:
                    tcc = TCC(
                        start_time=bars[i - cycle_length].timestamp,
                        end_time=bars[i].timestamp,
                        cycle_length=cycle_length,
                        cycle_type=cycle_type,
                        strength=strength
                    )
                    completions.append(tcc)

            # Check for local minimum (potential cycle trough)
            elif (closes[i] < closes[i-1] and closes[i] < closes[i+1] and
                  closes[i] < closes[i-2] and closes[i] < closes[i+2]):

                # Calculate cycle characteristics
                cycle_length = self._estimate_cycle_length(bars, i)
                cycle_type = self._classify_cycle_type(bars, i)
                strength = self._calculate_cycle_strength(bars, i)

                if cycle_length > 0 and strength > 3:
                    tcc = TCC(
                        start_time=bars[i - cycle_length].timestamp,
                        end_time=bars[i].timestamp,
                        cycle_length=cycle_length,
                        cycle_type=cycle_type,
                        strength=strength
                    )
                    completions.append(tcc)

        return completions

    def _estimate_cycle_length(self, bars: List[Bar], center_index: int) -> int:
        """Estimate the length of a cycle."""
        # Simplified cycle length estimation
        # Look for similar patterns in the past
        if center_index < 10:
            return 0

        # Use a simple approach: look for similar price movements
        center_price = bars[center_index].close

        # Look backwards for similar price levels
        for lookback in range(10, min(50, center_index)):
            past_price = bars[center_index - lookback].close
            price_ratio = abs(center_price - past_price) / past_price

            if price_ratio < 0.02:  # Within 2%
                return lookback

        return 0

    def _classify_cycle_type(self, bars: List[Bar], center_index: int) -> str:
        """Classify the type of cycle."""
        if center_index < 5:
            return "unknown"

        # Analyze price movement around the cycle center
        prices = [bar.close for bar in bars[center_index-5:center_index+6]]

        # Calculate slope before and after
        x_before = np.arange(5)
        x_after = np.arange(5, 10)
        y_before = prices[:5]
        y_after = prices[5:]

        slope_before = np.polyfit(x_before, y_before, 1)[0]
        slope_after = np.polyfit(x_after, y_after, 1)[0]

        # Classify based on slope changes
        if slope_before > 0 and slope_after < 0:
            return "impulse"
        elif slope_before < 0 and slope_after > 0:
            return "corrective"
        elif slope_before > 0 and slope_after > 0:
            return "trend_continuation"
        elif slope_before < 0 and slope_after < 0:
            return "trend_continuation_down"
        else:
            return "sideways"

    def _calculate_cycle_strength(self, bars: List[Bar], center_index: int) -> int:
        """Calculate the strength of a cycle."""
        if center_index < 5 or center_index >= len(bars) - 5:
            return 1

        # Calculate price movement magnitude
        center_price = bars[center_index].close
        max_deviation = 0

        for i in range(max(0, center_index - 5), min(len(bars), center_index + 6)):
            deviation = abs(bars[i].close - center_price) / center_price
            max_deviation = max(max_deviation, deviation)

        # Convert to strength score (1-10)
        strength = min(10, int(max_deviation * 500))
        return max(1, strength)

    def _analyze_trend_state(self, bars: List[Bar]) -> TrendState:
        """Analyze current trend state."""
        if len(bars) < self.trend_lookback:
            return TrendState.UNKNOWN

        closes = [bar.close for bar in bars]

        # Calculate trend indicators
        slope = self._calculate_trend_slope(closes)
        volatility = self._calculate_volatility(closes)

        # Determine trend state
        if abs(slope) < volatility * 0.1:
            return TrendState.RANGE
        elif slope > 0:
            return TrendState.TREND_UP
        else:
            return TrendState.TREND_DOWN

    def _determine_cycle_phase(self, market_state: MarketState, bars: List[Bar]) -> CyclePhase:
        """Determine current market cycle phase."""
        if len(bars) < self.trend_lookback:
            return CyclePhase.UNKNOWN

        # Use TCC and trend analysis to determine phase
        trend_state = self._analyze_trend_state(bars)
        current_tcc = market_state.current_tcc

        if trend_state == TrendState.TREND_UP:
            if current_tcc and current_tcc.cycle_type == "impulse":
                return CyclePhase.MARKUP
            else:
                return CyclePhase.ACCUMULATION
        elif trend_state == TrendState.TREND_DOWN:
            if current_tcc and current_tcc.cycle_type == "impulse":
                return CyclePhase.MARKDOWN
            else:
                return CyclePhase.DISTRIBUTION
        else:
            return CyclePhase.ACCUMULATION  # Default to accumulation in range

    def _identify_key_levels(self, market_state: MarketState, bars: List[Bar]) -> List[float]:
        """Identify key support/resistance levels."""
        if len(bars) < 20:
            return []

        levels = []
        highs = [bar.high for bar in bars]
        lows = [bar.low for bar in bars]

        # Find potential resistance levels (swing highs)
        for i in range(5, len(highs) - 5):
            if (highs[i] > highs[i-1] and highs[i] > highs[i+1] and
                highs[i] > highs[i-2] and highs[i] > highs[i+2]):
                levels.append(highs[i])

        # Find potential support levels (swing lows)
        for i in range(5, len(lows) - 5):
            if (lows[i] < lows[i-1] and lows[i] < lows[i+1] and
                lows[i] < lows[i-2] and lows[i] < lows[i+2]):
                levels.append(lows[i])

        # Remove duplicates and sort
        levels = sorted(list(set(levels)))

        # Return top 5 levels
        return levels[:5]

    def _calculate_trend_slope(self, prices: List[float]) -> float:
        """Calculate trend slope using linear regression."""
        x = np.arange(len(prices))
        y = np.array(prices)

        # Simple linear regression
        slope, _ = np.polyfit(x, y, 1)
        return slope

    def _calculate_volatility(self, prices: List[float]) -> float:
        """Calculate price volatility."""
        if len(prices) < 2:
            return 0.0

        returns = np.diff(prices) / prices[:-1]
        return np.std(returns) * np.sqrt(252)  # Annualized

    def get_trend_strength(self, market_state: MarketState) -> float:
        """Get current trend strength (0-1)."""
        if len(market_state.bars_5m) < self.trend_lookback:
            return 0.0

        recent_bars = market_state.get_latest_5m_bars(self.trend_lookback)
        closes = [bar.close for bar in recent_bars]

        # Calculate trend strength based on slope consistency
        slope = self._calculate_trend_slope(closes)
        volatility = self._calculate_volatility(closes)

        if volatility == 0:
            return 0.0

        # Normalize slope by volatility
        normalized_slope = abs(slope) / volatility
        return min(1.0, normalized_slope / 5.0)

    def get_cycle_confidence(self, market_state: MarketState) -> float:
        """Get confidence in current cycle analysis (0-1)."""
        if not market_state.current_tcc:
            return 0.0

        # Confidence based on TCC strength and recency
        tcc = market_state.current_tcc
        strength_score = tcc.strength / 10.0

        # Recency score
        if not market_state.bars_5m:
            return 0.0

        bars_since_tcc = len(market_state.bars_5m) - 1
        recency_score = max(0.0, 1.0 - (bars_since_tcc / 100.0))

        return (strength_score * 0.7 + recency_score * 0.3)

    def get_market_structure_summary(self, market_state: MarketState) -> Dict[str, Any]:
        """Get comprehensive market structure summary."""
        return {
            "trend_state": self._analyze_trend_state(market_state.get_latest_5m_bars(self.trend_lookback)).value,
            "cycle_phase": self._determine_cycle_phase(market_state, market_state.get_latest_5m_bars(self.trend_lookback)).value,
            "trend_strength": self.get_trend_strength(market_state),
            "cycle_confidence": self.get_cycle_confidence(market_state),
            "key_levels": self._identify_key_levels(market_state, market_state.get_latest_5m_bars(self.trend_lookback)),
            "current_tcc": market_state.current_tcc.model_dump() if market_state.current_tcc else None,
            "current_mcs": market_state.current_mcs.model_dump() if market_state.current_mcs else None
        }


# Numba-optimized functions
@jit(float64(float64[:]), nopython=True)
def calculate_trend_slope_numba(prices: np.ndarray) -> float64:
    """Numba-optimized trend slope calculation."""
    n = len(prices)
    if n < 2:
        return 0.0

    x = np.arange(n)
    sum_x = np.sum(x)
    sum_y = np.sum(prices)
    sum_xy = np.sum(x * prices)
    sum_x2 = np.sum(x * x)

    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
    return slope


@jit(float64(float64[:]), nopython=True)
def calculate_volatility_numba(prices: np.ndarray) -> float64:
    """Numba-optimized volatility calculation."""
    if len(prices) < 2:
        return 0.0

    returns = np.diff(prices) / prices[:-1]
    return np.std(returns) * np.sqrt(252.0)