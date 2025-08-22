"""
TCEA-SPMF v3 detector for PSE-LLM trading system.
Detects Smart Pullback & Micro-FVG (trend rhythm) setups.
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
    name="tcea_spmf_v3",
    config={
        "min_mqs": 7.0,
        "min_frs": 6.0,
        "min_tcc_strength": 0.6,  # Medium/High TCC required
        "min_retracement": 0.003,  # Minimum retracement (0.3%)
        "max_retracement": 0.02,   # Maximum retracement (2%)
        "min_micro_fvg_count": 1,
        "min_swing_strength": 6,
        "enabled": True
    },
    enabled=True
)
def detect_tcea_spmf_v3_setups(
    market_state: MarketState,
    bars_1m_window: List[Bar],
    config: Dict[str, Any]
) -> List[SetupProposal]:
    """
    Detect TCEA-SPMF v3 setups: Smart Pullback & Micro-FVG (trend rhythm).

    Args:
        market_state: Current market state
        bars_1m_window: 1-minute bars for context
        config: Detector configuration

    Returns:
        List of setup proposals
    """
    setups = []

    # Get configuration parameters
    min_mqs = config.get("min_mqs", 7.0)
    min_frs = config.get("min_frs", 6.0)
    min_tcc_strength = config.get("min_tcc_strength", 0.6)
    min_retracement = config.get("min_retracement", 0.003)
    max_retracement = config.get("max_retracement", 0.02)
    min_micro_fvg_count = config.get("min_micro_fvg_count", 1)
    min_swing_strength = config.get("min_swing_strength", 6)

    # Check if we have enough data
    if not market_state.bars_5m or len(market_state.bars_5m) < 30:
        return setups

    if not bars_1m_window or len(bars_1m_window) < 20:
        return setups

    # Check TCC condition (Medium/High trend required)
    if not market_state.current_tcc:
        return setups

    tcc_strength = market_state.current_tcc.strength / 10.0
    if tcc_strength < min_tcc_strength:
        return setups

    # Get recent market elements
    recent_swings = market_state.get_recent_swings(15)
    recent_1m_swings = market_state.get_recent_1m_swings(10)

    if not recent_swings:
        return setups

    # Identify impulse leg and pullback
    impulse_pullback_info = _identify_impulse_pullback(market_state, recent_swings, min_retracement, max_retracement)

    if not impulse_pullback_info["has_pattern"]:
        return setups

    # Check for micro-FVGs inside pullback
    micro_fvgs = _detect_micro_fvgs_in_pullback(impulse_pullback_info, bars_1m_window)

    if len(micro_fvgs) < min_micro_fvg_count:
        return setups

    # Check for breakout of pullback structure
    breakout_info = _check_pullback_breakout(impulse_pullback_info, market_state)

    if not breakout_info["has_breakout"]:
        return setups

    # Determine setup direction
    direction = impulse_pullback_info["direction"]

    # Calculate entry, SL, TP
    entry_price = breakout_info["breakout_price"]
    sl_price = _calculate_sl_price(impulse_pullback_info, direction)
    tp1_price = _calculate_tp1_price(entry_price, sl_price, direction)

    # Get relevant swings
    relevant_swings = _get_relevant_swings(recent_swings, impulse_pullback_info, min_swing_strength)

    # Create setup proposal
    setup = SetupProposal(
        symbol="ES",  # This should be configurable
        setup_type=SetupType.FVG,
        side=direction,
        entry_price=entry_price,
        stop_loss=sl_price,
        take_profit=tp1_price,
        risk_reward_ratio=_calculate_risk_reward(entry_price, sl_price, tp1_price),
        confidence=_calculate_confidence(tcc_strength, impulse_pullback_info["quality"], len(micro_fvgs)),
        swing_points=relevant_swings,
        mss_list=impulse_pullback_info["mss_list"],
        fvgs=micro_fvgs,
        volume_analysis={"impulse_volume_ratio": impulse_pullback_info["volume_ratio"]},
        order_flow={"breakout_strength": breakout_info["strength"]}
    )

    # Add evidence fields
    setup.evidence = {
        "tactic": "TCEA-SPMF v3",
        "mqs": min_mqs,  # Placeholder
        "frs": min_frs,  # Placeholder
        "tcc_strength": tcc_strength,
        "impulse_strength": impulse_pullback_info["impulse_strength"],
        "pullback_depth": impulse_pullback_info["retracement_pct"],
        "pullback_quality": impulse_pullback_info["quality"],
        "micro_fvg_count": len(micro_fvgs),
        "breakout_strength": breakout_info["strength"]
    }

    setups.append(setup)

    logger.debug(f"TCEA-SPMF v3 detector found {len(setups)} setups")
    return setups


def _identify_impulse_pullback(
    market_state: MarketState,
    swings: List[SwingPoint],
    min_retracement: float,
    max_retracement: float
) -> Dict[str, Any]:
    """Identify impulse leg and pullback pattern."""
    if len(swings) < 4:
        return {"has_pattern": False, "direction": "unknown"}

    # Look for impulse-pullback pattern in recent swings
    for i in range(len(swings) - 3):
        swing1 = swings[i]
        swing2 = swings[i + 1]
        swing3 = swings[i + 2]
        swing4 = swings[i + 3]

        # Check for bullish impulse-pullback
        if (swing1.swing_type == "SWING_LOW" and
            swing2.swing_type == "SWING_HIGH" and
            swing3.swing_type == "SWING_LOW" and
            swing4.swing_type == "SWING_HIGH"):

            # Calculate retracement
            impulse_high = swing2.price
            pullback_low = swing3.price
            retracement = (impulse_high - pullback_low) / impulse_high

            if min_retracement <= retracement <= max_retracement:
                # Get relevant MSS
                relevant_mss = _get_mss_for_swing_range(swing1.bar_index, swing4.bar_index, market_state)

                return {
                    "has_pattern": True,
                    "direction": Side.BUY,
                    "impulse_start": swing1.bar_index,
                    "impulse_end": swing2.bar_index,
                    "pullback_start": swing2.bar_index,
                    "pullback_end": swing3.bar_index,
                    "impulse_high": impulse_high,
                    "pullback_low": pullback_low,
                    "retracement_pct": retracement,
                    "impulse_strength": swing2.strength,
                    "quality": _calculate_pullback_quality(swing2, swing3, retracement),
                    "mss_list": relevant_mss,
                    "volume_ratio": _calculate_impulse_volume_ratio(swing1.bar_index, swing2.bar_index, market_state)
                }

        # Check for bearish impulse-pullback
        elif (swing1.swing_type == "SWING_HIGH" and
              swing2.swing_type == "SWING_LOW" and
              swing3.swing_type == "SWING_HIGH" and
              swing4.swing_type == "SWING_LOW"):

            # Calculate retracement
            impulse_low = swing2.price
            pullback_high = swing3.price
            retracement = (pullback_high - impulse_low) / impulse_low

            if min_retracement <= retracement <= max_retracement:
                # Get relevant MSS
                relevant_mss = _get_mss_for_swing_range(swing1.bar_index, swing4.bar_index, market_state)

                return {
                    "has_pattern": True,
                    "direction": Side.SELL,
                    "impulse_start": swing1.bar_index,
                    "impulse_end": swing2.bar_index,
                    "pullback_start": swing2.bar_index,
                    "pullback_end": swing3.bar_index,
                    "impulse_low": impulse_low,
                    "pullback_high": pullback_high,
                    "retracement_pct": retracement,
                    "impulse_strength": swing2.strength,
                    "quality": _calculate_pullback_quality(swing2, swing3, retracement),
                    "mss_list": relevant_mss,
                    "volume_ratio": _calculate_impulse_volume_ratio(swing1.bar_index, swing2.bar_index, market_state)
                }

    return {"has_pattern": False, "direction": "unknown"}


def _get_mss_for_swing_range(start_bar: int, end_bar: int, market_state: MarketState) -> List[MSS]:
    """Get MSS within a swing range."""
    relevant_mss = []

    for mss in market_state.get_recent_mss(10):
        if (mss.start_bar >= start_bar and mss.end_bar <= end_bar) and mss.is_valid:
            relevant_mss.append(mss)

    return relevant_mss


def _calculate_pullback_quality(swing_impulse: SwingPoint, swing_pullback: SwingPoint, retracement: float) -> float:
    """Calculate pullback quality (0-1)."""
    # Base quality from retracement (optimal is around 0.382-0.618)
    if 0.003 <= retracement <= 0.01:  # Optimal range
        retracement_score = 1.0
    elif 0.001 <= retracement <= 0.02:  # Acceptable range
        retracement_score = 0.7
    else:
        retracement_score = 0.3

    # Quality from swing strengths
    strength_score = min(1.0, (swing_impulse.strength + swing_pullback.strength) / 20.0)

    # Combined quality
    quality = (retracement_score * 0.6) + (strength_score * 0.4)

    return quality


def _calculate_impulse_volume_ratio(start_bar: int, end_bar: int, market_state: MarketState) -> float:
    """Calculate volume ratio during impulse."""
    if not market_state.bars_5m:
        return 1.0

    # Get volume during impulse
    start_idx = max(0, start_bar)
    end_idx = min(len(market_state.bars_5m), end_bar + 1)

    if start_idx >= end_idx:
        return 1.0

    impulse_volumes = [market_state.bars_5m[i].volume for i in range(start_idx, end_idx)]
    avg_impulse_volume = np.mean(impulse_volumes) if impulse_volumes else 0

    # Get average volume for comparison
    recent_volumes = [bar.volume for bar in market_state.get_latest_5m_bars(20)]
    avg_recent_volume = np.mean(recent_volumes) if recent_volumes else 1

    if avg_recent_volume == 0:
        return 1.0

    return avg_impulse_volume / avg_recent_volume


def _detect_micro_fvgs_in_pullback(
    impulse_pullback_info: Dict[str, Any],
    bars_1m_window: List[Bar]
) -> List[FVG]:
    """Detect micro-FVGs inside pullback."""
    micro_fvgs = []

    if not bars_1m_window:
        return micro_fvgs

    # Get pullback boundaries
    if impulse_pullback_info["direction"] == Side.BUY:
        pullback_high = impulse_pullback_info["impulse_high"]
        pullback_low = impulse_pullback_info["pullback_low"]
    else:
        pullback_high = impulse_pullback_info["pullback_high"]
        pullback_low = impulse_pullback_info["impulse_low"]

    # Look for micro-FVGs in recent 1m bars
    for i in range(2, len(bars_1m_window)):
        bar1 = bars_1m_window[i-2]
        bar2 = bars_1m_window[i-1]
        bar3 = bars_1m_window[i]

        # Check if bars are within pullback range
        if not (pullback_low <= min(bar1.low, bar2.low, bar3.low) <=
                max(bar1.high, bar2.high, bar3.high) <= pullback_high):
            continue

        # Check for bullish micro-FVG
        if bar1.high < bar3.low:
            fvg = FVG(
                start_bar=i-2,
                end_bar=i,
                fvg_type="BULLISH",
                top=bar3.low,
                bottom=bar1.high,
                is_filled=False
            )
            micro_fvgs.append(fvg)

        # Check for bearish micro-FVG
        elif bar1.low > bar3.high:
            fvg = FVG(
                start_bar=i-2,
                end_bar=i,
                fvg_type="BEARISH",
                top=bar1.low,
                bottom=bar3.high,
                is_filled=False
            )
            micro_fvgs.append(fvg)

    return micro_fvgs


def _check_pullback_breakout(
    impulse_pullback_info: Dict[str, Any],
    market_state: MarketState
) -> Dict[str, Any]:
    """Check for breakout of pullback structure."""
    if not market_state.bars_5m:
        return {"has_breakout": False, "strength": 0.0, "breakout_price": None}

    # Get recent bars
    recent_bars = market_state.get_latest_5m_bars(5)
    if len(recent_bars) < 3:
        return {"has_breakout": False, "strength": 0.0, "breakout_price": None}

    # Get pullback boundaries
    if impulse_pullback_info["direction"] == Side.BUY:
        resistance_level = impulse_pullback_info["impulse_high"]
        support_level = impulse_pullback_info["pullback_low"]

        # Check for bullish breakout
        latest_bar = recent_bars[-1]
        if latest_bar.close > resistance_level:
            breakout_strength = (latest_bar.close - resistance_level) / resistance_level
            return {
                "has_breakout": True,
                "strength": min(1.0, breakout_strength * 100),
                "breakout_price": latest_bar.close
            }

    else:  # SELL
        resistance_level = impulse_pullback_info["pullback_high"]
        support_level = impulse_pullback_info["impulse_low"]

        # Check for bearish breakout
        latest_bar = recent_bars[-1]
        if latest_bar.close < support_level:
            breakout_strength = (support_level - latest_bar.close) / support_level
            return {
                "has_breakout": True,
                "strength": min(1.0, breakout_strength * 100),
                "breakout_price": latest_bar.close
            }

    return {"has_breakout": False, "strength": 0.0, "breakout_price": None}


def _get_relevant_swings(
    swings: List[SwingPoint],
    impulse_pullback_info: Dict[str, Any],
    min_strength: int
) -> List[SwingPoint]:
    """Get relevant swing points for the setup."""
    relevant = []

    for swing in swings:
        # Check swing strength
        if swing.strength < min_strength:
            continue

        # Check if swing is part of the impulse-pullback pattern
        if (impulse_pullback_info["impulse_start"] <= swing.bar_index <= impulse_pullback_info["pullback_end"]):
            relevant.append(swing)

    return relevant


def _calculate_sl_price(impulse_pullback_info: Dict[str, Any], direction: Side) -> float:
    """Calculate stop loss price based on pullback extreme."""
    if direction == Side.BUY:
        # For bullish setup, SL below pullback low
        return impulse_pullback_info["pullback_low"] * 0.999  # Small buffer
    else:
        # For bearish setup, SL above pullback high
        return impulse_pullback_info["pullback_high"] * 1.001  # Small buffer


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


def _calculate_confidence(
    tcc_strength: float,
    pullback_quality: float,
    micro_fvg_count: int
) -> float:
    """Calculate setup confidence."""
    # Base confidence from TCC and pullback quality
    confidence = (tcc_strength * 0.4) + (pullback_quality * 0.4)

    # Boost for micro-FVG confirmations
    if micro_fvg_count >= 2:
        confidence += 0.2
    elif micro_fvg_count >= 1:
        confidence += 0.1

    return min(1.0, confidence)