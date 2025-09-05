"""
Quality kernel for PSE-LLM trading system.
Implements Market Quality Score (MQS) and Fair Value Gap Score (FRS) assessment.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

import numpy as np
from numba import float64, int64, jit

from engine.state import MarketState
from engine.trading_types import FVG, SetupProposal

logger = logging.getLogger(__name__)


class QualityGrade(str, Enum):
    """Quality assessment grades."""
    A_PLUS = "A+"
    A = "A"
    B = "B"
    C = "C"
    FAIL = "FAIL"


class QualityKernel:
    """Kernel for assessing market and setup quality."""

    def __init__(
        self,
        min_swing_strength: int = 7,
        max_fvg_staleness: float = 0.7,
        volume_threshold: float = 1.2
    ):
        """
        Initialize quality kernel.

        Args:
            min_swing_strength: Minimum swing strength for A grade
            max_fvg_staleness: Maximum FVG staleness for A grade
            volume_threshold: Minimum volume ratio for A grade
        """
        self.min_swing_strength = min_swing_strength
        self.max_fvg_staleness = max_fvg_staleness
        self.volume_threshold = volume_threshold

    def assess_setup_quality(self, setup: SetupProposal, market_state: MarketState) -> dict[str, Any]:
        """
        Assess overall quality of a setup proposal.

        Args:
            setup: Setup proposal to assess
            market_state: Current market state

        Returns:
            Dictionary with quality assessment results
        """
        # Calculate individual quality scores
        mqs = self.calculate_market_quality_score(market_state)
        frs = self.calculate_fair_value_gap_score(setup, market_state)

        # Calculate composite scores
        overall_score = (mqs * 0.6) + (frs * 0.4)  # Weighted average

        # Determine grade
        grade = self._score_to_grade(overall_score)

        # Check A+ criteria
        is_a_plus = self._check_a_plus_criteria(setup, market_state, mqs, frs)

        assessment = {
            "mqs": mqs,
            "frs": frs,
            "overall_score": overall_score,
            "grade": grade,
            "is_a_plus": is_a_plus,
            "assessment_details": {
                "swing_quality": self._assess_swing_quality(setup, market_state),
                "structure_quality": self._assess_structure_quality(setup, market_state),
                "volume_quality": self._assess_volume_quality(setup, market_state),
                "fvg_quality": self._assess_fvg_quality(setup, market_state)
            }
        }

        return assessment

    def calculate_market_quality_score(self, market_state: MarketState) -> float:
        """
        Calculate Market Quality Score (MQS) - overall market condition quality.

        Returns:
            Score between 0-10
        """
        if not market_state.bars_5m:
            return 0.0

        scores = []

        # Swing quality (30% weight)
        swing_score = self._calculate_swing_quality_score(market_state)
        scores.append(("swing", swing_score, 0.3))

        # Structure clarity (25% weight)
        structure_score = self._calculate_structure_clarity_score(market_state)
        scores.append(("structure", structure_score, 0.25))

        # Volume consistency (20% weight)
        volume_score = self._calculate_volume_consistency_score(market_state)
        scores.append(("volume", volume_score, 0.2))

        # Trend strength (15% weight)
        trend_score = self._calculate_trend_strength_score(market_state)
        scores.append(("trend", trend_score, 0.15))

        # Volatility appropriateness (10% weight)
        volatility_score = self._calculate_volatility_score(market_state)
        scores.append(("volatility", volatility_score, 0.1))

        # Calculate weighted average
        total_score = sum(score * weight for _, score, weight in scores)

        logger.debug(f"MQS breakdown: {scores}, Total: {total_score}")
        return total_score

    def calculate_fair_value_gap_score(self, setup: SetupProposal, market_state: MarketState) -> float:
        """
        Calculate Fair Value Gap Score (FRS) - quality of FVG involvement.

        Returns:
            Score between 0-10
        """
        if not setup.fvgs:
            return 0.0

        scores = []

        for fvg in setup.fvgs:
            fvg_score = 0.0

            # Size appropriateness (25% weight)
            size_score = self._assess_fvg_size(fvg, market_state)
            fvg_score += size_score * 0.25

            # Staleness (25% weight)
            staleness_score = self._assess_fvg_staleness(fvg, market_state)
            fvg_score += staleness_score * 0.25

            # Retest quality (25% weight)
            retest_score = self._assess_fvg_retest(fvg, market_state)
            fvg_score += retest_score * 0.25

            # Context alignment (25% weight)
            context_score = self._assess_fvg_context(fvg, setup, market_state)
            fvg_score += context_score * 0.25

            scores.append(fvg_score)

        # Return average FVG score
        return np.mean(scores) if scores else 0.0

    def _calculate_swing_quality_score(self, market_state: MarketState) -> float:
        """Calculate swing quality component of MQS."""
        if not market_state.swing_points:
            return 0.0

        recent_swings = market_state.get_recent_swings(10)
        if not recent_swings:
            return 0.0

        # Average swing strength
        avg_strength = np.mean([swing.strength for swing in recent_swings])

        # Distribution of swing strengths
        strong_swings = sum(1 for swing in recent_swings if swing.strength >= self.min_swing_strength)
        strength_distribution = strong_swings / len(recent_swings)

        # Recency of strong swings
        latest_swing_strength = recent_swings[-1].strength if recent_swings else 0

        # Combine factors
        score = (avg_strength / 10.0 * 0.4) + (strength_distribution * 0.4) + (latest_swing_strength / 10.0 * 0.2)
        return min(10.0, score * 10.0)

    def _calculate_structure_clarity_score(self, market_state: MarketState) -> float:
        """Calculate structure clarity component of MQS."""
        valid_mss = market_state.get_valid_mss()

        if not valid_mss:
            return 0.0

        # Number of valid MSS (more is better, up to a point)
        mss_count = len(valid_mss)
        count_score = min(10.0, mss_count * 2.0) / 10.0

        # MSS strength (based on price movement)
        latest_mss = valid_mss[-1]
        price_move = abs(latest_mss.confirmation_price - latest_mss.break_price)
        avg_price = (latest_mss.confirmation_price + latest_mss.break_price) / 2
        strength_score = min(10.0, (price_move / avg_price) * 1000) / 10.0

        # MSS recency (newer is better)
        bars_since_mss = len(market_state.bars_5m) - latest_mss.end_bar
        recency_score = max(0.0, 1.0 - (bars_since_mss / 50.0))

        return (count_score * 0.3 + strength_score * 0.4 + recency_score * 0.3) * 10.0

    def _calculate_volume_consistency_score(self, market_state: MarketState) -> float:
        """Calculate volume consistency component of MQS."""
        if len(market_state.bars_5m) < 10:
            return 0.0

        recent_bars = market_state.get_latest_5m_bars(20)
        volumes = [bar.volume for bar in recent_bars]

        # Coefficient of variation (lower is more consistent)
        mean_volume = np.mean(volumes)
        std_volume = np.std(volumes)

        if mean_volume == 0:
            return 0.0

        cv = std_volume / mean_volume
        consistency_score = max(0.0, 1.0 - cv)

        # Volume trend (slightly increasing is good)
        volume_trend = np.polyfit(range(len(volumes)), volumes, 1)[0]
        trend_score = min(1.0, max(0.0, volume_trend * 1000 + 0.5))

        return (consistency_score * 0.7 + trend_score * 0.3) * 10.0

    def _calculate_trend_strength_score(self, market_state: MarketState) -> float:
        """Calculate trend strength component of MQS."""
        if len(market_state.bars_5m) < 20:
            return 0.0

        recent_bars = market_state.get_latest_5m_bars(20)
        closes = [float(bar.close) for bar in recent_bars]

        # Linear regression slope
        x = np.arange(len(closes))
        slope, _ = np.polyfit(x, closes, 1)

        # Normalize slope by price level
        avg_price = np.mean(closes)
        normalized_slope = slope / avg_price

        # Convert to score (strong trend in either direction)
        trend_strength = min(10.0, abs(normalized_slope) * 10000)
        return trend_strength

    def _calculate_volatility_score(self, market_state: MarketState) -> float:
        """Calculate volatility appropriateness component of MQS."""
        if len(market_state.bars_5m) < 10:
            return 5.0  # Neutral score

        recent_bars = market_state.get_latest_5m_bars(20)
        closes = [float(bar.close) for bar in recent_bars]

        # Calculate returns
        returns = np.diff(closes) / closes[:-1]
        volatility = np.std(returns) * np.sqrt(252)  # Annualized

        # Score based on volatility level (moderate volatility is best)
        if volatility < 0.1:  # Low volatility
            return 3.0
        elif volatility < 0.2:  # Moderate volatility
            return 8.0
        elif volatility < 0.4:  # High volatility
            return 6.0
        else:  # Very high volatility
            return 2.0

    def _assess_fvg_size(self, fvg: FVG, market_state: MarketState) -> float:
        """Assess FVG size appropriateness."""
        gap_size = fvg.top - fvg.bottom
        avg_price = (fvg.top + fvg.bottom) / 2

        if avg_price == 0:
            return 0.0

        normalized_size = gap_size / avg_price

        # Score based on size (moderate gaps are best)
        if normalized_size < 0.001:  # Very small
            return 2.0
        elif normalized_size < 0.005:  # Small
            return 8.0
        elif normalized_size < 0.01:  # Medium
            return 6.0
        else:  # Large
            return 3.0

    def _assess_fvg_staleness(self, fvg: FVG, market_state: MarketState) -> float:
        """Assess FVG staleness."""
        if not market_state.bars_5m:
            return 0.0

        current_bar = len(market_state.bars_5m) - 1
        bars_since_creation = current_bar - fvg.end_bar

        # Staleness score (newer is better)
        staleness = min(1.0, bars_since_creation / 100.0)
        return (1.0 - staleness) * 10.0

    def _assess_fvg_retest(self, fvg: FVG, market_state: MarketState) -> float:
        """Assess FVG retest quality."""
        # This would need access to structure kernel for retest info
        # For now, return a neutral score
        return 5.0

    def _assess_fvg_context(self, fvg: FVG, setup: SetupProposal, market_state: MarketState) -> float:
        """Assess FVG context alignment with setup."""
        # Check if FVG aligns with setup direction
        if setup.side == "BUY" and fvg.fvg_type == "BULLISH":
            return 8.0
        elif setup.side == "SELL" and fvg.fvg_type == "BEARISH":
            return 8.0
        else:
            return 3.0

    def _score_to_grade(self, score: float) -> QualityGrade:
        """Convert numerical score to quality grade."""
        if score >= 9.0:
            return QualityGrade.A_PLUS
        elif score >= 8.0:
            return QualityGrade.A
        elif score >= 6.0:
            return QualityGrade.B
        elif score >= 4.0:
            return QualityGrade.C
        else:
            return QualityGrade.FAIL

    def _check_a_plus_criteria(self, setup: SetupProposal, market_state: MarketState, mqs: float, frs: float) -> bool:
        """Check if setup meets A+ criteria."""
        criteria = []

        # MQS must be A+ level
        criteria.append(mqs >= 9.0)

        # FRS must be A level
        criteria.append(frs >= 8.0)

        # Must have strong swing involvement
        strong_swings = [s for s in setup.swing_points if s.strength >= self.min_swing_strength]
        criteria.append(len(strong_swings) >= 2)

        # Must have valid MSS
        criteria.append(len(setup.mss_list) > 0)

        # Volume must be above threshold
        criteria.append(self._check_volume_threshold(setup, market_state))

        return all(criteria)

    def _check_volume_threshold(self, setup: SetupProposal, market_state: MarketState) -> bool:
        """Check if volume meets threshold."""
        # This would need volume data from the setup
        # For now, return True
        return True

    def _assess_swing_quality(self, setup: SetupProposal, market_state: MarketState) -> dict[str, Any]:
        """Assess swing quality for the setup."""
        if not setup.swing_points:
            return {"score": 0.0, "details": "No swing points"}

        avg_strength = np.mean([s.strength for s in setup.swing_points])
        strong_count = sum(1 for s in setup.swing_points if s.strength >= self.min_swing_strength)

        return {
            "score": min(10.0, avg_strength),
            "avg_strength": avg_strength,
            "strong_count": strong_count,
            "total_count": len(setup.swing_points)
        }

    def _assess_structure_quality(self, setup: SetupProposal, market_state: MarketState) -> dict[str, Any]:
        """Assess structure quality for the setup."""
        mss_count = len(setup.mss_list)
        valid_mss = [mss for mss in setup.mss_list if mss.is_valid]

        return {
            "score": min(10.0, mss_count * 2.0),
            "mss_count": mss_count,
            "valid_mss_count": len(valid_mss)
        }

    def _assess_volume_quality(self, setup: SetupProposal, market_state: MarketState) -> dict[str, Any]:
        """Assess volume quality for the setup."""
        # This would need volume analysis from setup.evidence
        return {
            "score": 5.0,  # Neutral score
            "details": "Volume analysis not implemented"
        }

    def _assess_fvg_quality(self, setup: SetupProposal, market_state: MarketState) -> dict[str, Any]:
        """Assess FVG quality for the setup."""
        if not setup.fvgs:
            return {"score": 0.0, "details": "No FVGs"}

        active_fvgs = [fvg for fvg in setup.fvgs if not fvg.is_filled]

        return {
            "score": min(10.0, len(active_fvgs) * 3.0),
            "active_fvgs": len(active_fvgs),
            "total_fvgs": len(setup.fvgs)
        }


# Numba-optimized quality calculations
@jit(float64(float64[:], int64), nopython=True)
def calculate_trend_strength_numba(prices: np.ndarray, window: int) -> float64:
    """Numba-optimized trend strength calculation."""
    if len(prices) < window:
        return 0.0

    # Simple linear regression slope
    x = np.arange(len(prices))
    y = prices

    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x * x)

    n = len(x)
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)

    # Normalize by average price
    avg_price = np.mean(prices)
    if avg_price == 0:
        return 0.0

    normalized_slope = slope / avg_price
    return min(10.0, abs(normalized_slope) * 10000.0)


@jit(float64(float64[:]), nopython=True)
def calculate_volatility_score_numba(returns: np.ndarray) -> float64:
    """Numba-optimized volatility scoring."""
    if len(returns) == 0:
        return 5.0

    volatility = np.std(returns) * np.sqrt(252.0)  # Annualized

    if volatility < 0.1:
        return 3.0
    elif volatility < 0.2:
        return 8.0
    elif volatility < 0.4:
        return 6.0
    else:
        return 2.0
