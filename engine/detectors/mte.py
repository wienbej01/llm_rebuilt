"""
TCEA-MTE detector for PSE-LLM trading system.
Detects Momentum Thrust Entry setups.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import logging

import numpy as np

from engine.types import SetupProposal, Bar, Side, SetupType, SwingPoint, MSS, FVG
from engine.state import MarketState
from engine.detectors.registry import register_detector

logger = logging.getLogger(__name__)


@register_detector(
    name="mte",
    config={
        "min_mqs": 8.0,
        "min_frs": 7.0,
        "min_tcc_strength": 0.8,  # High TCC required
        "min_thrust_strength": 0.015,  # 1.5% minimum thrust
        "min_swing_strength": 7,
        "min_volume_ratio": 1.5,
        "enabled": True
    },
    enabled=True
)
def detect_mte_setups(
    market_state: MarketState,
    bars_1m_window: List[Bar],
    config: Dict[str, Any]
) -> List[SetupProposal]:
    """
    Detect TCEA-MTE setups: Momentum Thrust Entry.

    Args:
        market_state: Current market state
        bars_1m_window: 1-minute bars for context
        config: Detector configuration

    Returns:
        List of setup proposals
    """
    setups = []

    # Get configuration parameters
    min_mqs = config.get("min_mqs", 8.0)
    min_frs = config.get("min_frs", 7.0)
    min_tcc_strength = config.get("min_tcc_strength", 0.8)
    min_thrust_strength = config.get("min_thrust_strength", 0.015)
    min_swing_strength = config.get("min_swing_strength", 7)
    min_volume_ratio = config.get("min_volume_ratio", 1.5)

    # Check if we have enough data
    if not market_state.bars_5m or len(market_state.bars_5m) < 20:
        return setups

    if not bars_1m_window or len(bars_1m_window) < 10:
        return setups

    # Check TCC condition (High trend required)
    if not market_state.current_tcc:
        return setups

    tcc_strength = market_state.current_tcc.strength / 10.0
    if tcc_strength < min_tcc_strength:
        return setups

    # Get recent market elements
    recent_mss = market_state.get_recent_mss(5)
    recent_swings = market_state.get_recent_swings(10)

    if not recent_mss:
        return setups

    # Look for momentum thrust patterns
    for mss in recent_mss:
        if not mss.is_valid:
            continue

        # Check if MSS has high MQS
        mss_quality = _assess_mss_quality(mss, market_state, min_swing_strength, min_volume_ratio)
        if mss_quality < min_mqs:
            continue

        # Check for thrust pattern
        thrust_info = _identify_thrust_pattern(mss, market_state, min_thrust_strength)
        if not thrust_info["has_thrust"]:
            continue

        # Determine setup direction
        direction = Side.BUY if mss.direction == "BULLISH" else Side.SELL

        # Calculate entry, SL, TP
        entry_price = _calculate_entry_price(thrust_info, bars_1m_window, direction)
        sl_price = _calculate_sl_price(thrust_info, direction)
        tp1_price = _calculate_tp1_price(entry_price, sl_price, direction)

        # Get relevant swings
        relevant_swings = _get_relevant_swings(recent_swings, mss, min_swing_strength)

        # Create setup proposal
        setup = SetupProposal(
            symbol="ES",  # This should be configurable
            setup_type=SetupType.CHANGE_OF_CHARACTER,
            side=direction,
            entry_price=entry_price,
            stop_loss=sl_price,
            take_profit=tp1_price,
            risk_reward_ratio=_calculate_risk_reward(entry_price, sl_price, tp1_price),
            confidence=_calculate_confidence(tcc_strength, mss_quality, thrust_info["strength"]),
            swing_points=relevant_swings,
            mss_list=[mss],
            fvgs=[],  # MTE typically doesn't rely on FVGs
            volume_analysis={"thrust_volume_ratio": thrust_info["volume_ratio"]},
            order_flow={"thrust_strength": thrust_info["strength"]}
        )

        # Add evidence fields
        setup.evidence = {
            "tactic": "TCEA-MTE",
            "mqs": mss_quality,
            "frs": mss_quality,
            "tcc_strength": tcc_strength,
            "thrust_strength": thrust_info["strength"],
            "thrust_direction": thrust_info["direction"],
            "breakout_level": thrust_info["breakout_level"],
            "thrust_volume_ratio": thrust_info["volume_ratio"],
            "entry_timing": thrust_info["entry_timing"]
        }

        setups.append(setup)

    logger.debug(f"TCEA-MTE detector found {len(setups)} setups")
    return setups


def _assess_mss_quality(
    mss: MSS,
    market_state: MarketState,
    min_swing_strength: int,
    min_volume_ratio: float
) -> float:
    """Assess MSS quality (0-10)."""
    quality = 0.0

    # Base quality from MSS strength
    quality += min(3.0, mss.direction == "BULLISH" or mss.direction == "BEARISH") * 3.0

    # Quality from supporting swings
    supporting_swings = _get_supporting_swings(mss, market_state, min_swing_strength)
    swing_score = min(3.0, len(supporting_swings) * 1.0)
    quality += swing_score

    # Quality from volume confirmation
    volume_ratio = _calculate_mss_volume_ratio(mss, market_state)
    if volume_ratio >= min_volume_ratio * 1.5:  # Strong volume
        quality += 4.0
    elif volume_ratio >= min_volume_ratio:  # Good volume
        quality += 2.0
    elif volume_ratio >= min_volume_ratio * 0.7:  # Acceptable volume
        quality += 1.0

    return min(10.0, quality)


def _get_supporting_swings(
    mss: MSS,
    market_state: MarketState,
    min_strength: int
) -> List[SwingPoint]:
    """Get swings that support the MSS."""
    supporting_swings = []

    recent_swings = market_state.get_recent_swings(15)

    for swing in recent_swings:
        if swing.strength < min_strength:
            continue

        # Check if swing supports MSS direction
        if (mss.direction == "BULLISH" and swing.swing_type == "SWING_LOW") or \
           (mss.direction == "BEARISH" and swing.swing_type == "SWING_HIGH"):
            # Check if swing is near MSS
            if abs(swing.bar_index - mss.end_bar) <= 5:
                supporting_swings.append(swing)

    return supporting_swings


def _calculate_mss_volume_ratio(mss: MSS, market_state: MarketState) -> float:
    """Calculate volume ratio around MSS creation."""
    if not market_state.bars_5m:
        return 1.0

    # Get volume around MSS creation
    start_idx = max(0, mss.start_bar - 2)
    end_idx = min(len(market_state.bars_5m), mss.end_bar + 3)

    if start_idx >= end_idx:
        return 1.0

    mss_volumes = [market_state.bars_5m[i].volume for i in range(start_idx, end_idx)]
    avg_mss_volume = np.mean(mss_volumes) if mss_volumes else 0

    # Get average volume for comparison
    recent_volumes = [bar.volume for bar in market_state.get_latest_5m_bars(20)]
    avg_recent_volume = np.mean(recent_volumes) if recent_volumes else 1

    if avg_recent_volume == 0:
        return 1.0

    return avg_mss_volume / avg_recent_volume


def _identify_thrust_pattern(
    mss: MSS,
    market_state: MarketState,
    min_strength: float
) -> Dict[str, Any]:
    """Identify thrust pattern after MSS."""
    if not market_state.bars_5m:
        return {"has_thrust": False, "strength": 0.0}

    # Look for thrust in bars after MSS confirmation
    start_idx = mss.end_bar + 1
    end_idx = min(len(market_state.bars_5m), mss.end_bar + 5)

    if start_idx >= end_idx:
        return {"has_thrust": False, "strength": 0.0}

    thrust_bars = market_state.bars_5m[start_idx:end_idx]

    # Calculate thrust metrics
    if mss.direction == "BULLISH":
        # Look for upward thrust
        thrust_strength = _calculate_bullish_thrust(thrust_bars, mss.confirmation_price)
        breakout_level = mss.confirmation_price
        thrust_direction = "bullish"
    else:
        # Look for downward thrust
        thrust_strength = _calculate_bearish_thrust(thrust_bars, mss.confirmation_price)
        breakout_level = mss.confirmation_price
        thrust_direction = "bearish"

    # Calculate volume ratio
    volume_ratio = _calculate_thrust_volume_ratio(thrust_bars, market_state)

    # Determine entry timing
    entry_timing = _determine_entry_timing(thrust_bars, thrust_direction)

    return {
        "has_thrust": thrust_strength >= min_strength,
        "strength": min(1.0, thrust_strength / 0.05),  # Normalize to 0-1
        "direction": thrust_direction,
        "breakout_level": breakout_level,
        "volume_ratio": volume_ratio,
        "entry_timing": entry_timing,
        "thrust_bars": thrust_bars
    }


def _calculate_bullish_thrust(bars: List[Bar], base_price: float) -> float:
    """Calculate bullish thrust strength."""
    if not bars:
        return 0.0

    # Find maximum upward movement
    max_close = max(bar.close for bar in bars)
    thrust_strength = (max_close - base_price) / base_price

    return thrust_strength


def _calculate_bearish_thrust(bars: List[Bar], base_price: float) -> float:
    """Calculate bearish thrust strength."""
    if not bars:
        return 0.0

    # Find maximum downward movement
    min_close = min(bar.close for bar in bars)
    thrust_strength = (base_price - min_close) / base_price

    return thrust_strength


def _calculate_thrust_volume_ratio(thrust_bars: List[Bar], market_state: MarketState) -> float:
    """Calculate volume ratio during thrust."""
    if not thrust_bars:
        return 1.0

    # Get volume during thrust
    thrust_volumes = [bar.volume for bar in thrust_bars]
    avg_thrust_volume = np.mean(thrust_volumes) if thrust_volumes else 0

    # Get average volume for comparison
    recent_volumes = [bar.volume for bar in market_state.get_latest_5m_bars(20)]
    avg_recent_volume = np.mean(recent_volumes) if recent_volumes else 1

    if avg_recent_volume == 0:
        return 1.0

    return avg_thrust_volume / avg_recent_volume


def _determine_entry_timing(thrust_bars: List[Bar], thrust_direction: str) -> str:
    """Determine optimal entry timing."""
    if not thrust_bars:
        return "unknown"

    # Check if thrust is still strong
    if len(thrust_bars) >= 2:
        latest_bar = thrust_bars[-1]
        previous_bar = thrust_bars[-2]

        if thrust_direction == "bullish":
            if latest_bar.close > previous_bar.close:
                return "immediate"  # Enter at first 1m of next 5m
            else:
                return "pullback"  # Wait for pullback
        else:  # bearish
            if latest_bar.close < previous_bar.close:
                return "immediate"  # Enter at first 1m of next 5m
            else:
                return "pullback"  # Wait for pullback

    return "immediate"


def _calculate_entry_price(
    thrust_info: Dict[str, Any],
    bars_1m_window: List[Bar],
    direction: Side
) -> float:
    """Calculate entry price based on thrust pattern."""
    if thrust_info["entry_timing"] == "immediate":
        # Enter at first 1m of next 5m period
        if bars_1m_window:
            return bars_1m_window[-1].close
        else:
            return thrust_info["breakout_level"]
    else:
        # Enter on pullback (simplified)
        if direction == Side.BUY:
            return thrust_info["breakout_level"] * 0.999  # Small discount
        else:
            return thrust_info["breakout_level"] * 1.001  # Small premium


def _calculate_sl_price(thrust_info: Dict[str, Any], direction: Side) -> float:
    """Calculate stop loss price based on thrust pattern."""
    if direction == Side.BUY:
        # For bullish setup, SL below thrust start
        return thrust_info["breakout_level"] * 0.995  # 0.5% buffer
    else:
        # For bearish setup, SL above thrust start
        return thrust_info["breakout_level"] * 1.005  # 0.5% buffer


def _calculate_tp1_price(entry_price: float, sl_price: float, direction: Side) -> float:
    """Calculate take profit 1 price."""
    risk_points = abs(entry_price - sl_price)
    reward_points = risk_points * 1.5  # 1:1.5 risk:reward ratio

    if direction == Side.BUY:
        return entry_price + reward_points
    else:
        return entry_price - reward_points


def _calculate_risk_reward(entry_price: float, sl_price: float, tp_price: float) -> float:
    """Calculate risk:reward ratio."""
    risk = abs(entry_price - sl_price)
    reward = abs(tp_price - entry_price)

    if risk == 0:
        return 0.0

    return reward / risk


def _get_relevant_swings(
    swings: List[SwingPoint],
    mss: MSS,
    min_strength: int
) -> List[SwingPoint]:
    """Get relevant swing points for the setup."""
    relevant = []

    for swing in swings:
        # Check swing strength
        if swing.strength < min_strength:
            continue

        # Check if swing is related to MSS
        if (min(swing.bar_index, mss.start_bar) <= max(swing.bar_index, mss.end_bar)):
            relevant.append(swing)

    return relevant


def _calculate_confidence(
    tcc_strength: float,
    mss_quality: float,
    thrust_strength: float
) -> float:
    """Calculate setup confidence."""
    # Base confidence from TCC, MSS quality, and thrust strength
    confidence = (tcc_strength * 0.3) + (mss_quality / 10.0 * 0.4) + (thrust_strength * 0.3)

    return min(1.0, confidence)