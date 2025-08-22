"""
Micro 1m kernel for PSE-LLM trading system.
Analyzes 1-minute bars for micro-structure clues and confirmation signals.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import logging

import numpy as np
from numba import jit, float64, int64

from engine.types import Bar, SwingPoint, FVG
from engine.state import MarketState

logger = logging.getLogger(__name__)


class MicroSignalType(str, Enum):
    """Micro signal type enumeration."""
    MICRO_PULLBACK = "micro_pullback"
    MICRO_FVG = "micro_fvg"
    MICRO_SWING = "micro_swing"
    VOLUME_SPIKE = "volume_spike"
    PRICE_REJECTION = "price_rejection"
    MOMENTUM_SHIFT = "momentum_shift"


class Micro1mKernel:
    """Kernel for 1-minute micro-structure analysis."""

    def __init__(
        self,
        micro_lookback: int = 20,
        volume_threshold: float = 2.0,
        price_threshold: float = 0.001
    ):
        """
        Initialize micro 1m kernel.

        Args:
            micro_lookback: Number of 1m bars to look back for micro analysis
            volume_threshold: Volume spike threshold (multiple of average)
            price_threshold: Price movement threshold for micro signals
        """
        self.micro_lookback = micro_lookback
        self.volume_threshold = volume_threshold
        self.price_threshold = price_threshold

    def analyze_micro_structure(self, market_state: MarketState, bars_1m: List[Bar]) -> Dict[str, Any]:
        """
        Analyze 1-minute micro-structure for confirmation clues.

        Args:
            market_state: Current market state
            bars_1m: List of 1-minute bars for analysis

        Returns:
            Dictionary with micro-structure analysis results
        """
        if len(bars_1m) < self.micro_lookback:
            return {"signals": [], "summary": "Insufficient data for micro analysis"}

        # Add 1m bars to state
        for bar in bars_1m:
            market_state.add_1m_bar(bar)

        # Detect micro signals
        micro_signals = []

        # Micro pullbacks
        pullbacks = self._detect_micro_pullbacks(bars_1m)
        micro_signals.extend(pullbacks)

        # Micro FVGs
        micro_fvgs = self._detect_micro_fvgs(bars_1m)
        micro_signals.extend(micro_fvgs)

        # Micro swings
        micro_swings = self._detect_micro_swings(bars_1m)
        micro_signals.extend(micro_swings)

        # Volume spikes
        volume_spikes = self._detect_volume_spikes(bars_1m)
        micro_signals.extend(volume_spikes)

        # Price rejection
        price_rejections = self._detect_price_rejection(bars_1m)
        micro_signals.extend(price_rejections)

        # Momentum shifts
        momentum_shifts = self._detect_momentum_shifts(bars_1m)
        micro_signals.extend(momentum_shifts)

        # Analyze signal quality and alignment
        signal_analysis = self._analyze_signal_quality(micro_signals, market_state)

        result = {
            "signals": micro_signals,
            "signal_count": len(micro_signals),
            "analysis": signal_analysis,
            "summary": self._generate_micro_summary(micro_signals, signal_analysis),
            "timestamp": datetime.now(timezone.utc)
        }

        logger.debug(f"Micro analysis: {len(micro_signals)} signals detected")
        return result

    def _detect_micro_pullbacks(self, bars_1m: List[Bar]) -> List[Dict[str, Any]]:
        """Detect micro pullbacks in 1-minute data."""
        if len(bars_1m) < 10:
            return []

        pullbacks = []
        closes = [bar.close for bar in bars_1m]

        # Simple pullback detection: retracement against recent trend
        for i in range(5, len(closes) - 5):
            # Determine local trend
            recent_trend = self._calculate_local_trend(closes[i-5:i])

            if abs(recent_trend) < 0.001:  # No clear trend
                continue

            # Look for pullback (counter-trend movement)
            current_trend = self._calculate_local_trend(closes[i-2:i+3])

            if (recent_trend > 0 and current_trend < -0.001) or (recent_trend < 0 and current_trend > 0.001):
                pullback = {
                    "type": MicroSignalType.MICRO_PULLBACK,
                    "bar_index": i,
                    "timestamp": bars_1m[i].timestamp,
                    "price": bars_1m[i].close,
                    "depth": abs(current_trend),
                    "context_trend": recent_trend,
                    "strength": self._calculate_pullback_strength(bars_1m, i)
                }
                pullbacks.append(pullback)

        return pullbacks

    def _detect_micro_fvgs(self, bars_1m: List[Bar]) -> List[Dict[str, Any]]:
        """Detect micro Fair Value Gaps in 1-minute data."""
        if len(bars_1m) < 3:
            return []

        micro_fvgs = []

        for i in range(2, len(bars_1m)):
            bar1, bar2, bar3 = bars_1m[i-2], bars_1m[i-1], bars_1m[i]

            # Check for bullish micro FVG
            if bar1.high < bar3.low:
                gap_size = (bar3.low - bar1.high) / bar1.high
                if gap_size > self.price_threshold:
                    micro_fvg = {
                        "type": MicroSignalType.MICRO_FVG,
                        "bar_index": i,
                        "timestamp": bar3.timestamp,
                        "top": bar3.low,
                        "bottom": bar1.high,
                        "size": gap_size,
                        "direction": "bullish",
                        "is_filled": False
                    }
                    micro_fvgs.append(micro_fvg)

            # Check for bearish micro FVG
            elif bar1.low > bar3.high:
                gap_size = (bar1.low - bar3.high) / bar1.low
                if gap_size > self.price_threshold:
                    micro_fvg = {
                        "type": MicroSignalType.MICRO_FVG,
                        "bar_index": i,
                        "timestamp": bar3.timestamp,
                        "top": bar1.low,
                        "bottom": bar3.high,
                        "size": gap_size,
                        "direction": "bearish",
                        "is_filled": False
                    }
                    micro_fvgs.append(micro_fvg)

        return micro_fvgs

    def _detect_micro_swings(self, bars_1m: List[Bar]) -> List[Dict[str, Any]]:
        """Detect micro swing points in 1-minute data."""
        if len(bars_1m) < 5:
            return []

        micro_swings = []

        for i in range(2, len(bars_1m) - 2):
            current_bar = bars_1m[i]

            # Check for micro swing high
            if (current_bar.high > bars_1m[i-1].high and
                current_bar.high > bars_1m[i+1].high and
                current_bar.high > bars_1m[i-2].high and
                current_bar.high > bars_1m[i+2].high):

                micro_swing = {
                    "type": MicroSignalType.MICRO_SWING,
                    "bar_index": i,
                    "timestamp": current_bar.timestamp,
                    "price": current_bar.high,
                    "swing_type": "high",
                    "strength": self._calculate_micro_swing_strength(bars_1m, i, "high")
                }
                micro_swings.append(micro_swing)

            # Check for micro swing low
            elif (current_bar.low < bars_1m[i-1].low and
                  current_bar.low < bars_1m[i+1].low and
                  current_bar.low < bars_1m[i-2].low and
                  current_bar.low < bars_1m[i+2].low):

                micro_swing = {
                    "type": MicroSignalType.MICRO_SWING,
                    "bar_index": i,
                    "timestamp": current_bar.timestamp,
                    "price": current_bar.low,
                    "swing_type": "low",
                    "strength": self._calculate_micro_swing_strength(bars_1m, i, "low")
                }
                micro_swings.append(micro_swing)

        return micro_swings

    def _detect_volume_spikes(self, bars_1m: List[Bar]) -> List[Dict[str, Any]]:
        """Detect volume spikes in 1-minute data."""
        if len(bars_1m) < 10:
            return []

        volume_spikes = []
        volumes = [bar.volume for bar in bars_1m]

        # Calculate average volume
        avg_volume = np.mean(volumes[-10:])

        for i in range(len(bars_1m)):
            if volumes[i] > avg_volume * self.volume_threshold:
                volume_spike = {
                    "type": MicroSignalType.VOLUME_SPIKE,
                    "bar_index": i,
                    "timestamp": bars_1m[i].timestamp,
                    "volume": volumes[i],
                    "avg_volume": avg_volume,
                    "multiplier": volumes[i] / avg_volume,
                    "price_action": self._analyze_price_action_at_spike(bars_1m, i)
                }
                volume_spikes.append(volume_spike)

        return volume_spikes

    def _detect_price_rejection(self, bars_1m: List[Bar]) -> List[Dict[str, Any]]:
        """Detect price rejection patterns in 1-minute data."""
        if len(bars_1m) < 3:
            return []

        price_rejections = []

        for i in range(1, len(bars_1m) - 1):
            prev_bar = bars_1m[i-1]
            current_bar = bars_1m[i]
            next_bar = bars_1m[i+1]

            # Check for rejection patterns (long wicks, reversals)
            rejection_signals = []

            # Upper rejection (long upper wick)
            if (current_bar.high - current_bar.close) / (current_bar.high - current_bar.low) > 0.6:
                rejection_signals.append("upper_wick")

            # Lower rejection (long lower wick)
            if (current_bar.close - current_bar.low) / (current_bar.high - current_bar.low) > 0.6:
                rejection_signals.append("lower_wick")

            # Reversal pattern
            if (prev_bar.close > prev_bar.open and
                current_bar.close < current_bar.open and
                next_bar.close > next_bar.open):
                rejection_signals.append("bullish_reversal")

            elif (prev_bar.close < prev_bar.open and
                  current_bar.close > current_bar.open and
                  next_bar.close < next_bar.open):
                rejection_signals.append("bearish_reversal")

            if rejection_signals:
                price_rejection = {
                    "type": MicroSignalType.PRICE_REJECTION,
                    "bar_index": i,
                    "timestamp": current_bar.timestamp,
                    "signals": rejection_signals,
                    "price": current_bar.close,
                    "range": current_bar.high - current_bar.low
                }
                price_rejections.append(price_rejection)

        return price_rejections

    def _detect_momentum_shifts(self, bars_1m: List[Bar]) -> List[Dict[str, Any]]:
        """Detect momentum shifts in 1-minute data."""
        if len(bars_1m) < 10:
            return []

        momentum_shifts = []
        closes = [bar.close for bar in bars_1m]

        for i in range(5, len(closes)):
            # Calculate momentum before and after
            momentum_before = self._calculate_momentum(closes[i-5:i])
            momentum_after = self._calculate_momentum(closes[i-4:i+1])

            # Check for significant momentum shift
            momentum_change = abs(momentum_after - momentum_before)
            if momentum_change > 0.002:  # 0.2% threshold
                momentum_shift = {
                    "type": MicroSignalType.MOMENTUM_SHIFT,
                    "bar_index": i,
                    "timestamp": bars_1m[i].timestamp,
                    "momentum_before": momentum_before,
                    "momentum_after": momentum_after,
                    "change": momentum_change,
                    "direction": "bullish" if momentum_after > momentum_before else "bearish"
                }
                momentum_shifts.append(momentum_shift)

        return momentum_shifts

    def _calculate_local_trend(self, prices: List[float]) -> float:
        """Calculate local trend over a small window."""
        if len(prices) < 2:
            return 0.0

        # Simple linear regression slope
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)

        # Normalize by average price
        avg_price = np.mean(prices)
        if avg_price == 0:
            return 0.0

        return slope / avg_price

    def _calculate_pullback_strength(self, bars_1m: List[Bar], index: int) -> float:
        """Calculate pullback strength."""
        if index < 3 or index >= len(bars_1m) - 3:
            return 0.0

        # Look at price movement around pullback
        prices = [bar.close for bar in bars_1m[index-3:index+4]]
        price_range = max(prices) - min(prices)
        avg_price = np.mean(prices)

        if avg_price == 0:
            return 0.0

        return min(1.0, price_range / avg_price * 100)

    def _calculate_micro_swing_strength(self, bars_1m: List[Bar], index: int, swing_type: str) -> float:
        """Calculate micro swing strength."""
        if index < 2 or index >= len(bars_1m) - 2:
            return 0.0

        swing_price = bars_1m[index].high if swing_type == "high" else bars_1m[index].low

        # Count how many bars the swing dominates
        dominance_count = 0
        for i in range(max(0, index - 3), min(len(bars_1m), index + 4)):
            if i == index:
                continue

            if swing_type == "high":
                if bars_1m[i].high < swing_price:
                    dominance_count += 1
            else:
                if bars_1m[i].low > swing_price:
                    dominance_count += 1

        return min(1.0, dominance_count / 6.0)

    def _analyze_price_action_at_spike(self, bars_1m: List[Bar], index: int) -> str:
        """Analyze price action at volume spike."""
        if index == 0 or index >= len(bars_1m):
            return "unknown"

        bar = bars_1m[index]
        prev_bar = bars_1m[index-1]

        # Determine price action type
        if bar.close > bar.open and bar.close > prev_bar.close:
            return "bullish_breakout"
        elif bar.close < bar.open and bar.close < prev_bar.close:
            return "bearish_breakout"
        elif abs(bar.close - bar.open) / (bar.high - bar.low) < 0.3:
            return "indecision"
        else:
            return "mixed"

    def _calculate_momentum(self, prices: List[float]) -> float:
        """Calculate momentum over a window."""
        if len(prices) < 2:
            return 0.0

        # Simple momentum calculation
        returns = np.diff(prices) / prices[:-1]
        return np.mean(returns)

    def _analyze_signal_quality(self, signals: List[Dict[str, Any]], market_state: MarketState) -> Dict[str, Any]:
        """Analyze quality and alignment of micro signals."""
        if not signals:
            return {"quality": 0.0, "alignment": "none", "confidence": 0.0}

        # Count signal types
        signal_types = {}
        for signal in signals:
            signal_type = signal["type"]
            signal_types[signal_type] = signal_types.get(signal_type, 0) + 1

        # Calculate quality score
        quality_score = min(10.0, len(signals) * 2.0)

        # Determine alignment
        bullish_signals = 0
        bearish_signals = 0

        for signal in signals:
            if signal["type"] == MicroSignalType.MICRO_PULLBACK:
                # Context dependent
                continue
            elif signal["type"] == MicroSignalType.MICRO_FVG:
                if signal["direction"] == "bullish":
                    bullish_signals += 1
                else:
                    bearish_signals += 1
            elif signal["type"] == MicroSignalType.MICRO_SWING:
                if signal["swing_type"] == "high":
                    bearish_signals += 1
                else:
                    bullish_signals += 1
            elif signal["type"] == MicroSignalType.VOLUME_SPIKE:
                if signal["price_action"] == "bullish_breakout":
                    bullish_signals += 1
                elif signal["price_action"] == "bearish_breakout":
                    bearish_signals += 1

        # Determine overall alignment
        if bullish_signals > bearish_signals * 2:
            alignment = "bullish"
        elif bearish_signals > bullish_signals * 2:
            alignment = "bearish"
        else:
            alignment = "mixed"

        # Calculate confidence
        total_directional = bullish_signals + bearish_signals
        confidence = total_directional / len(signals) if signals else 0.0

        return {
            "quality": quality_score,
            "alignment": alignment,
            "confidence": confidence,
            "signal_types": signal_types,
            "bullish_count": bullish_signals,
            "bearish_count": bearish_signals
        }

    def _generate_micro_summary(self, signals: List[Dict[str, Any]], analysis: Dict[str, Any]) -> str:
        """Generate summary of micro analysis."""
        if not signals:
            return "No significant micro signals detected"

        summary_parts = []

        # Signal count
        summary_parts.append(f"{len(signals)} micro signals detected")

        # Quality
        quality = analysis["quality"]
        if quality >= 8.0:
            summary_parts.append("high quality")
        elif quality >= 6.0:
            summary_parts.append("moderate quality")
        else:
            summary_parts.append("low quality")

        # Alignment
        alignment = analysis["alignment"]
        if alignment != "mixed":
            summary_parts.append(f"{alignment} bias")

        return ", ".join(summary_parts)

    def get_micro_confirmation_score(self, market_state: MarketState, direction: str) -> float:
        """Get micro confirmation score for a given direction."""
        if not market_state.bars_1m:
            return 0.0

        recent_bars_1m = market_state.get_latest_1m_bars(self.micro_lookback)
        micro_analysis = self.analyze_micro_structure(market_state, recent_bars_1m)

        analysis = micro_analysis["analysis"]

        if direction == "bullish":
            if analysis["alignment"] == "bullish":
                return analysis["confidence"] * analysis["quality"] / 10.0
            elif analysis["alignment"] == "bearish":
                return -analysis["confidence"] * analysis["quality"] / 10.0
        else:  # bearish
            if analysis["alignment"] == "bearish":
                return analysis["confidence"] * analysis["quality"] / 10.0
            elif analysis["alignment"] == "bullish":
                return -analysis["confidence"] * analysis["quality"] / 10.0

        return 0.0


# Numba-optimized functions
@jit(float64(float64[:]), nopython=True)
def calculate_local_trend_numba(prices: np.ndarray) -> float64:
    """Numba-optimized local trend calculation."""
    n = len(prices)
    if n < 2:
        return 0.0

    x = np.arange(n)
    sum_x = np.sum(x)
    sum_y = np.sum(prices)
    sum_xy = np.sum(x * prices)
    sum_x2 = np.sum(x * x)

    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
    avg_price = np.mean(prices)

    if avg_price == 0:
        return 0.0

    return slope / avg_price


@jit(float64(float64[:]), nopython=True)
def calculate_momentum_numba(prices: np.ndarray) -> float64:
    """Numba-optimized momentum calculation."""
    if len(prices) < 2:
        return 0.0

    returns = np.diff(prices) / prices[:-1]
    return np.mean(returns)