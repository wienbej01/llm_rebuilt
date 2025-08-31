"""
TCEA-FVG-MR detector for PSE-LLM trading system.
Detects Missed FVG → Mean Reversion (re-entry) setups.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from engine.detectors.registry import register_detector
from engine.state import MarketState
from engine.trading_types import FVG, Bar, SetupProposal, SetupType, Side, SwingPoint

logger = logging.getLogger(__name__)


@register_detector(
    name="tcea_fvg_mr",
    config={
        "min_mqs": 6.0,
        "min_frs": 6.0,
        "min_fvg_width": 0.001,
        "max_fvg_staleness": 0.8,
        "min_reaction_strength": 0.01,  # 1% minimum reaction
        "min_1m_swing_strength": 5,
        "max_sl_points": 8.0,
        "enabled": True
    },
    enabled=True
)
def detect_tcea_fvg_mr_setups(
    market_state: MarketState,
    bars_1m_window: list[Bar],
    config: dict[str, Any]
) -> list[SetupProposal]:
    """
    Detect TCEA-FVG-MR setups: Missed FVG → Mean Reversion (re-entry).

    Args:
        market_state: Current market state
        bars_1m_window: 1-minute bars for context
        config: Detector configuration

    Returns:
        List of setup proposals
    """
    setups = []

    # Get configuration parameters
    min_mqs = config.get("min_mqs", 6.0)
    min_frs = config.get("min_frs", 6.0)
    min_fvg_width = config.get("min_fvg_width", 0.001)
    max_fvg_staleness = config.get("max_fvg_staleness", 0.8)
    min_reaction_strength = config.get("min_reaction_strength", 0.01)
    min_1m_swing_strength = config.get("min_1m_swing_strength", 5)
    max_sl_points = config.get("max_sl_points", 8.0)

    # Check if we have enough data
    if not market_state.bars_5m or len(market_state.bars_5m) < 30:
        return setups

    if not bars_1m_window or len(bars_1m_window) < 20:
        return setups

    # Get recent market elements
    recent_fvgs = market_state.get_recent_fvgs(15)
    recent_1m_swings = market_state.get_recent_1m_swings(10)

    if not recent_fvgs or not recent_1m_swings:
        return setups

    # Look for missed FVGs with strong reactions
    for fvg in recent_fvgs:
        if fvg.is_filled:
            continue

        # Check if FVG was missed (POI never filled)
        if not _is_fvg_missed(fvg, market_state):
            continue

        # Check if there was a strong reaction
        reaction_info = _check_fvg_reaction(fvg, market_state, min_reaction_strength)
        if not reaction_info["has_reaction"]:
            continue

        # Check for 1m pullback with MSS confirmation
        pullback_info = _check_1m_pullback_mss(fvg, market_state, bars_1m_window, min_1m_swing_strength)
        if not pullback_info["has_pullback"]:
            continue

        # Determine setup direction
        direction = _determine_setup_direction(fvg, reaction_info["reaction_direction"])

        # Calculate entry, SL, TP
        entry_price = pullback_info["breakout_price"]
        sl_price = _calculate_sl_price(pullback_info, direction, max_sl_points)
        tp1_price = _calculate_tp1_price(entry_price, sl_price, direction)

        # Calculate sizing multiplier based on reaction strength
        sizing_multiplier = _calculate_sizing_multiplier(reaction_info["strength"])

        # Create setup proposal
        setup = SetupProposal(
            symbol="ES",  # This should be configurable
            setup_type=SetupType.FVG,
            side=direction,
            entry_price=entry_price,
            stop_loss=sl_price,
            take_profit=tp1_price,
            risk_reward_ratio=_calculate_risk_reward(entry_price, sl_price, tp1_price),
            confidence=_calculate_confidence(reaction_info["strength"], pullback_info["strength"]),
            swing_points=pullback_info["swing_points"],
            mss_list=pullback_info["mss_list"],
            fvgs=[fvg],
            volume_analysis={"reaction_volume_ratio": reaction_info["volume_ratio"]},
            order_flow={"pullback_strength": pullback_info["strength"]}
        )

        # Add evidence fields
        setup.evidence = {
            "tactic": "TCEA-FVG-MR",
            "mqs": min_mqs,  # Placeholder
            "frs": min_frs,  # Placeholder
            "fvg_width": (fvg.top - fvg.bottom) / ((fvg.top + fvg.bottom) / 2),
            "fvg_staleness": _calculate_fvg_staleness(fvg, market_state),
            "reaction_strength": reaction_info["strength"],
            "reaction_distance": reaction_info["distance"],
            "pullback_depth": pullback_info["depth"],
            "sizing_multiplier": sizing_multiplier
        }

        setups.append(setup)

    logger.debug(f"TCEA-FVG-MR detector found {len(setups)} setups")
    return setups


def _is_fvg_missed(fvg: FVG, market_state: MarketState) -> bool:
    """Check if FVG POI was never filled."""
    if not market_state.bars_5m:
        return False

    poi_price = (fvg.top + fvg.bottom) / 2  # Point of Interest

    # Check if POI was ever touched
    for i in range(fvg.end_bar, len(market_state.bars_5m)):
        bar = market_state.bars_5m[i]

        if bar.low <= poi_price <= bar.high:
            return False  # POI was touched, not missed

    return True  # POI was never touched


def _check_fvg_reaction(
    fvg: FVG,
    market_state: MarketState,
    min_strength: float
) -> dict[str, Any]:
    """Check if there was a strong reaction to the FVG."""
    if not market_state.bars_5m:
        return {"has_reaction": False, "strength": 0.0, "distance": 0.0, "direction": "unknown"}

    poi_price = (fvg.top + fvg.bottom) / 2

    # Look for reaction in bars after FVG creation
    max_reaction_distance = 0.0
    reaction_direction = "unknown"
    reaction_volume_ratio = 1.0

    for i in range(fvg.end_bar + 1, len(market_state.bars_5m)):
        bar = market_state.bars_5m[i]

        # Calculate distance from POI
        if fvg.fvg_type == "BULLISH":
            distance = (bar.close - poi_price) / poi_price
            if distance > max_reaction_distance:
                max_reaction_distance = distance
                reaction_direction = "bullish"
        else:  # BEARISH
            distance = (poi_price - bar.close) / poi_price
            if distance > max_reaction_distance:
                max_reaction_distance = distance
                reaction_direction = "bearish"

    # Calculate volume ratio during reaction
    if max_reaction_distance >= min_strength:
        reaction_volume_ratio = _calculate_reaction_volume_ratio(fvg, market_state)

    return {
        "has_reaction": max_reaction_distance >= min_strength,
        "strength": min(1.0, max_reaction_distance / 0.05),  # Normalize to 0-1
        "distance": max_reaction_distance,
        "direction": reaction_direction,
        "volume_ratio": reaction_volume_ratio
    }


def _calculate_reaction_volume_ratio(fvg: FVG, market_state: MarketState) -> float:
    """Calculate volume ratio during FVG reaction."""
    if not market_state.bars_5m:
        return 1.0

    # Get volume during reaction period
    start_idx = fvg.end_bar + 1
    end_idx = min(len(market_state.bars_5m), fvg.end_bar + 10)

    if start_idx >= end_idx:
        return 1.0

    reaction_volumes = [market_state.bars_5m[i].volume for i in range(start_idx, end_idx)]
    avg_reaction_volume = np.mean(reaction_volumes) if reaction_volumes else 0

    # Get average volume for comparison
    recent_volumes = [bar.volume for bar in market_state.get_latest_5m_bars(20)]
    avg_recent_volume = np.mean(recent_volumes) if recent_volumes else 1

    if avg_recent_volume == 0:
        return 1.0

    return avg_reaction_volume / avg_recent_volume


def _check_1m_pullback_mss(
    fvg: FVG,
    market_state: MarketState,
    bars_1m_window: list[Bar],
    min_swing_strength: int
) -> dict[str, Any]:
    """Check for 1m pullback with MSS confirmation."""
    if not bars_1m_window:
        return {"has_pullback": False, "strength": 0.0, "breakout_price": None}

    # Get recent 1m swings
    recent_1m_swings = market_state.get_recent_1m_swings(10)

    # Look for pullback pattern
    pullback_info = _identify_pullback_pattern(bars_1m_window, recent_1m_swings, fvg)

    if not pullback_info["has_pullback"]:
        return {"has_pullback": False, "strength": 0.0, "breakout_price": None}

    # Check for MSS confirmation
    mss_confirmation = _check_mss_confirmation(pullback_info, market_state)

    return {
        "has_pullback": True,
        "strength": pullback_info["strength"],
        "depth": pullback_info["depth"],
        "breakout_price": pullback_info["breakout_price"],
        "swing_points": pullback_info["swing_points"],
        "mss_list": mss_confirmation["mss_list"]
    }


def _identify_pullback_pattern(
    bars_1m_window: list[Bar],
    swings_1m: list[SwingPoint],
    fvg: FVG
) -> dict[str, Any]:
    """Identify pullback pattern in 1m data."""
    if len(bars_1m_window) < 10:
        return {"has_pullback": False, "strength": 0.0}

    # Determine expected direction based on FVG
    expected_direction = "bullish" if fvg.fvg_type == "BULLISH" else "bearish"

    # Look for pullback against expected direction
    closes = [bar.close for bar in bars_1m_window[-10:]]

    # Simple pullback detection
    if expected_direction == "bullish":
        # Look for downward pullback
        pullback_start = max(closes)
        pullback_end = min(closes[-5:])  # Recent lows
        pullback_depth = (pullback_start - pullback_end) / pullback_start

        # Check for breakout
        breakout_price = closes[-1]
        breakout_strength = (breakout_price - pullback_end) / pullback_end if pullback_end > 0 else 0

    else:  # bearish
        # Look for upward pullback
        pullback_start = min(closes)
        pullback_end = max(closes[-5:])  # Recent highs
        pullback_depth = (pullback_end - pullback_start) / pullback_start

        # Check for breakout
        breakout_price = closes[-1]
        breakout_strength = (pullback_end - breakout_price) / pullback_end if pullback_end > 0 else 0

    # Get relevant swings
    relevant_swings = [s for s in swings_1m if s.strength >= 5]

    return {
        "has_pullback": pullback_depth > 0.002 and breakout_strength > 0.001,
        "strength": min(1.0, breakout_strength * 100),
        "depth": pullback_depth,
        "breakout_price": breakout_price,
        "swing_points": relevant_swings
    }


def _check_mss_confirmation(pullback_info: dict[str, Any], market_state: MarketState) -> dict[str, Any]:
    """Check for MSS confirmation of pullback."""
    recent_mss = market_state.get_recent_mss(5)

    # Look for MSS that confirms the direction
    confirming_mss = []
    for mss in recent_mss:
        if mss.is_valid:
            confirming_mss.append(mss)

    return {
        "mss_list": confirming_mss
    }


def _determine_setup_direction(fvg: FVG, reaction_direction: str) -> Side:
    """Determine setup direction based on FVG and reaction."""
    if fvg.fvg_type == "BULLISH":
        # Bullish FVG with bullish reaction = continuation
        # Bullish FVG with bearish reaction = mean reversion
        return Side.BUY if reaction_direction == "bearish" else Side.SELL
    else:  # BEARISH
        # Bearish FVG with bearish reaction = continuation
        # Bearish FVG with bullish reaction = mean reversion
        return Side.SELL if reaction_direction == "bullish" else Side.BUY


def _calculate_sl_price(pullback_info: dict[str, Any], direction: Side, max_sl_points: float) -> float:
    """Calculate stop loss price based on pullback."""
    if direction == Side.BUY:
        # For bullish setup, SL below pullback low
        pullback_low = pullback_info["breakout_price"] * (1 - pullback_info["depth"])
        sl_points = min(max_sl_points, pullback_info["breakout_price"] - pullback_low)
        return pullback_info["breakout_price"] - sl_points
    else:
        # For bearish setup, SL above pullback high
        pullback_high = pullback_info["breakout_price"] * (1 + pullback_info["depth"])
        sl_points = min(max_sl_points, pullback_high - pullback_info["breakout_price"])
        return pullback_info["breakout_price"] + sl_points


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


def _calculate_sizing_multiplier(reaction_strength: float) -> float:
    """Calculate sizing multiplier based on reaction strength."""
    if reaction_strength >= 0.8:  # Very strong reaction
        return 1.0
    elif reaction_strength >= 0.5:  # Strong reaction
        return 0.8
    elif reaction_strength >= 0.3:  # Moderate reaction
        return 0.6
    else:  # Weak reaction
        return 0.5


def _calculate_confidence(reaction_strength: float, pullback_strength: float) -> float:
    """Calculate setup confidence."""
    # Base confidence from reaction and pullback strength
    confidence = (reaction_strength * 0.6) + (pullback_strength * 0.4)

    return min(1.0, confidence)


def _calculate_fvg_staleness(fvg: FVG, market_state: MarketState) -> float:
    """Calculate FVG staleness (0-1, where 1 is most stale)."""
    if not market_state.bars_5m:
        return 1.0

    current_bar = len(market_state.bars_5m) - 1
    bars_since_creation = current_bar - fvg.end_bar

    return min(1.0, bars_since_creation / 100.0)
