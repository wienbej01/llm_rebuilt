"""
DAS 2.2 detector for PSE-LLM trading system.
Detects Liquidity Sweep & Reversal (post-sweep DAS 2.1) setups.
"""

from __future__ import annotations
from decimal import Decimal

from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import logging

import numpy as np

from engine.types import SetupProposal, Bar, Side, SetupType, SwingPoint, MSS, FVG
from engine.state import MarketState
from engine.detectors.registry import register_detector

logger = logging.getLogger(__name__)


@register_detector(
    name="das22_sweep",
    config={
        "min_mqs": 7.0,
        "min_frs": 6.0,
        "min_fvg_width": 0.001,
        "max_fvg_staleness": 0.7,
        "retest_tolerance": 0.0005,
        "min_swing_strength": 6,
        "sweep_lookback": 20,
        "min_sweep_strength": 0.002,
        "enabled": True
    },
    enabled=True
)
def detect_das22_setups(
    market_state: MarketState,
    bars_1m_window: List[Bar],
    config: Dict[str, Any]
) -> List[SetupProposal]:
    """
    Detect DAS 2.2 setups: Liquidity Sweep & Reversal (post-sweep DAS 2.1).

    Args:
        market_state: Current market state
        bars_1m_window: 1-minute bars for context
        config: Detector configuration

    Returns:
        List of setup proposals
    """
    setups = []

    # Get configuration parameters
    min_mqs = float(config.get("min_mqs", 7.0))
    min_frs = float(config.get("min_frs", 6.0))
    min_fvg_width = Decimal(str(config.get("min_fvg_width", 0.001)))
    max_fvg_staleness = float(config.get("max_fvg_staleness", 0.7))
    retest_tolerance = Decimal(str(config.get("retest_tolerance", 0.0005)))
    min_swing_strength = config.get("min_swing_strength", 6)
    sweep_lookback = int(config.get("sweep_lookback", 20))
    min_sweep_strength = Decimal(str(config.get("min_sweep_strength", 0.002)))

    # Check if we have enough data
    if not market_state.bars_5m or len(market_state.bars_5m) < 30:
        return setups

    # Get recent market elements
    recent_mss = market_state.get_recent_mss(5)
    recent_fvgs = market_state.get_recent_fvgs(10)
    recent_swings = market_state.get_recent_swings(20)

    if not recent_mss or not recent_fvgs or not recent_swings:
        return setups

    # Detect liquidity sweeps
    sweeps = _detect_liquidity_sweeps(market_state, recent_swings, sweep_lookback, min_sweep_strength)

    # Look for DAS 2.2 setups (sweep + reversal)
    for sweep in sweeps:
        # Find MSS that confirms reversal
        reversal_mss = _find_reversal_mss(sweep, recent_mss)
        if not reversal_mss:
            continue

        # Find FVGs created after sweep
        post_sweep_fvgs = _find_post_sweep_fvgs(sweep, recent_fvgs)
        if not post_sweep_fvgs:
            continue

        # Process each valid FVG
        for fvg in post_sweep_fvgs:
            # Check FVG quality
            fvg_quality = _assess_fvg_quality(fvg, market_state, min_fvg_width, max_fvg_staleness)
            if fvg_quality < min_frs:
                continue

            # Check for retest opportunity
            retest_info = _check_fvg_retest(fvg, market_state, retest_tolerance)
            if not retest_info["has_retest"]:
                continue

            # Determine setup direction (opposite to sweep)
            direction = Side.BUY if sweep["direction"] == "bearish" else Side.SELL

            # Get relevant swings
            relevant_swings = _get_relevant_swings(recent_swings, reversal_mss, fvg, min_swing_strength)

            # Calculate entry, SL, TP
            entry_price = retest_info["retest_price"]
            sl_price = _calculate_sl_price(fvg, direction)
            tp1_price = _calculate_tp1_price(entry_price, sl_price, direction)

            # Create setup proposal
            setup = SetupProposal(
                symbol="ES",  # This should be configurable
                setup_type=SetupType.LIQUIDITY_SWEEP,
                side=direction,
                entry_price=entry_price,
                stop_loss=sl_price,
                take_profit=tp1_price,
                risk_reward_ratio=_calculate_risk_reward(entry_price, sl_price, tp1_price),
                confidence=_calculate_confidence(fvg_quality, len(relevant_swings), sweep["strength"]),
                swing_points=relevant_swings,
                mss_list=[reversal_mss],
                fvgs=[fvg],
                volume_analysis={"fvg_volume_ratio": _calculate_fvg_volume_ratio(fvg, market_state)},
                order_flow={"retest_strength": retest_info["strength"]}
            )

            # Add evidence fields with sweep metadata
            setup.evidence = {
                "tactic": "DAS 2.2",
                "mqs": fvg_quality,
                "frs": fvg_quality,
                "fvg_width": (fvg.top - fvg.bottom) / ((fvg.top + fvg.bottom) / 2),
                "fvg_staleness": _calculate_fvg_staleness(fvg, market_state),
                "retest_distance": abs(entry_price - ((fvg.top + fvg.bottom) / 2)),
                "swing_strength": np.mean([s.strength for s in relevant_swings]) if relevant_swings else 0,
                "sweep_meta": {
                    "sweep_price": sweep["price"],
                    "sweep_direction": sweep["direction"],
                    "sweep_strength": sweep["strength"],
                    "sweep_type": sweep["type"],
                    "sweep_bar": sweep["bar_index"]
                }
            }

            setups.append(setup)

    logger.debug(f"DAS 2.2 detector found {len(setups)} setups")
    return setups


def _detect_liquidity_sweeps(
    market_state: MarketState,
    swings: List[SwingPoint],
    lookback: int,
    min_strength: float
) -> List[Dict[str, Any]]:
    """Detect liquidity sweeps in recent price action."""
    sweeps = []

    if not market_state.bars_5m or len(market_state.bars_5m) < lookback:
        return sweeps

    recent_bars = market_state.get_latest_5m_bars(lookback)

    # Get key levels from swings
    key_levels = []
    for swing in swings:
        if swing.strength >= 7:  # Only strong swings
            key_levels.append({
                "price": swing.price,
                "type": swing.swing_type.value,
                "bar_index": swing.bar_index,
                "strength": swing.strength
            })

    # Look for sweeps of key levels
    for i in range(5, len(recent_bars)):
        bar = recent_bars[i]

        for level in key_levels:
            # Check if bar swept the level
            if _is_level_swept(bar, level, min_strength):
                sweep = {
                    "bar_index": len(market_state.bars_5m) - len(recent_bars) + i,
                    "price": level["price"],
                    "type": level["type"],
                    "direction": "bullish" if level["type"] == "SWING_LOW" else "bearish",
                    "strength": _calculate_sweep_strength(bar, level),
                    "level_strength": level["strength"],
                    "swept_by": bar
                }
                sweeps.append(sweep)

    return sweeps


def _is_level_swept(bar: Bar, level: Dict[str, Any], min_strength: Decimal) -> bool:
    """Check if a bar swept a key level."""
    level_price = level["price"]
    level_type = level["type"]

    if level_type == "SWING_HIGH":
        # Bearish sweep: price closes below swing high
        return bar.close < level_price and (bar.high - level_price) / level_price >= min_strength
    else:  # SWING_LOW
        # Bullish sweep: price closes above swing low
        return bar.close > level_price and (level_price - bar.low) / level_price >= min_strength


def _calculate_sweep_strength(bar: Bar, level: Dict[str, Any]) -> float:
    """Calculate sweep strength (0-1)."""
    level_price = level["price"]
    level_type = level["type"]

    if level_type == "SWING_HIGH":
        sweep_distance = bar.high - level_price
    else:  # SWING_LOW
        sweep_distance = level_price - bar.low

    # Normalize by level price
    normalized_distance = sweep_distance / level_price

    # Strength based on distance and volume
    volume_factor = min(2.0, bar.volume / 1000.0)  # Normalize volume

    return min(1.0, normalized_distance * 100 * volume_factor)


def _find_reversal_mss(sweep: Dict[str, Any], mss_list: List[MSS]) -> Optional[MSS]:
    """Find MSS that confirms reversal after sweep."""
    sweep_bar = sweep["bar_index"]
    sweep_direction = sweep["direction"]

    # Look for MSS after sweep
    for mss in mss_list:
        if mss.start_bar < sweep_bar:
            continue  # MSS before sweep

        # Check if MSS confirms reversal
        if (sweep_direction == "bearish" and mss.direction == "BULLISH") or \
           (sweep_direction == "bullish" and mss.direction == "BEARISH"):
            return mss

    return None


def _find_post_sweep_fvgs(sweep: Dict[str, Any], fvg_list: List[FVG]) -> List[FVG]:
    """Find FVGs created after sweep."""
    sweep_bar = sweep["bar_index"]
    post_sweep_fvgs = []

    for fvg in fvg_list:
        if fvg.start_bar >= sweep_bar and not fvg.is_filled:
            post_sweep_fvgs.append(fvg)

    return post_sweep_fvgs


def _assess_fvg_quality(
    fvg: FVG,
    market_state: MarketState,
    min_width: Decimal,
    max_staleness: float
) -> float:
    """Assess FVG quality (0-10)."""
    quality = 0.0

    # Size quality (0-3 points)
    fvg_width = (fvg.top - fvg.bottom) / ((fvg.top + fvg.bottom) / Decimal('2'))
    if min_width * Decimal('2') <= fvg_width <= min_width * Decimal('5'):  # Optimal size
        quality += 3.0
    elif fvg_width >= min_width:  # Acceptable size
        quality += 2.0
    elif fvg_width >= min_width * Decimal('0.5'):  # Small but acceptable
        quality += 1.0

    # Staleness quality (0-3 points)
    staleness = _calculate_fvg_staleness(fvg, market_state)
    if float(staleness) <= max_staleness * 0.3:  # Very fresh
        quality += 3.0
    elif float(staleness) <= max_staleness * 0.7:  # Fresh
        quality += 2.0
    elif float(staleness) <= max_staleness:  # Acceptable
        quality += 1.0

    # Retest quality (0-4 points)
    retest_info = _check_fvg_retest(fvg, market_state, Decimal('0.001'))
    if retest_info["has_retest"]:
        if retest_info["strength"] >= 0.8:  # Strong retest
            quality += 4.0
        elif retest_info["strength"] >= 0.5:  # Good retest
            quality += 3.0
        else:  # Weak retest
            quality += 1.0

    return min(10.0, quality)


def _calculate_fvg_staleness(fvg: FVG, market_state: MarketState) -> Decimal:
    """Calculate FVG staleness (0-1, where 1 is most stale)."""
    if not market_state.bars_5m:
        return Decimal('1.0')

    current_bar = len(market_state.bars_5m) - 1
    bars_since_creation = current_bar - fvg.end_bar

    return min(Decimal('1.0'), Decimal(bars_since_creation) / Decimal('100.0'))


def _check_fvg_retest(
    fvg: FVG,
    market_state: MarketState,
    tolerance: Decimal
) -> Dict[str, Any]:
    """Check if FVG has been retested and return retest info."""
    if not market_state.bars_5m:
        return {"has_retest": False, "retest_price": None, "strength": 0.0}

    poi_price = (fvg.top + fvg.bottom) / 2  # Point of Interest

    # Look for retest in recent bars
    for i in range(fvg.start_bar, len(market_state.bars_5m)):
        bar = market_state.bars_5m[i]

        # Check if price touched POI
        if bar.low <= poi_price <= bar.high:
            # Calculate retest strength
            strength = _calculate_retest_strength(bar, fvg, poi_price)

            return {
                "has_retest": True,
                "retest_price": poi_price,
                "strength": strength,
                "retest_bar": i
            }

    return {"has_retest": False, "retest_price": None, "strength": 0.0}


def _calculate_retest_strength(bar: Bar, fvg: FVG, poi_price: Decimal) -> float:
    """Calculate retest strength (0-1)."""
    # Perfect retest is when price closes exactly at POI
    close_distance = abs(bar.close - poi_price)
    fvg_range = fvg.top - fvg.bottom

    if fvg_range == Decimal('0'):
        return 0.0

    # Normalize distance by FVG range
    normalized_distance = close_distance / fvg_range

    # Strength decreases with distance
    strength = float(max(Decimal('0.0'), Decimal('1.0') - normalized_distance * Decimal('10')))

    return strength


def _get_relevant_swings(
    swings: List[SwingPoint],
    mss: MSS,
    fvg: FVG,
    min_strength: int
) -> List[SwingPoint]:
    """Get relevant swing points for the setup."""
    relevant = []

    for swing in swings:
        # Check swing strength
        if swing.strength < min_strength:
            continue

        # Check if swing is related to MSS or FVG
        if (min(swing.bar_index, mss.start_bar) <= max(swing.bar_index, mss.end_bar) or
            min(swing.bar_index, fvg.start_bar) <= max(swing.bar_index, fvg.end_bar)):
            relevant.append(swing)

    return relevant


def _calculate_sl_price(fvg: FVG, direction: Side) -> Decimal:
    """Calculate stop loss price based on FVG."""
    if direction == Side.BUY:
        # For bullish setup, SL at bottom of FVG
        return fvg.bottom
    else:
        # For bearish setup, SL at top of FVG
        return fvg.top


def _calculate_tp1_price(entry_price: Decimal, sl_price: Decimal, direction: Side) -> Decimal:
    """Calculate take profit 1 price."""
    risk_points = abs(entry_price - sl_price)
    reward_points = risk_points * Decimal('1.5')  # 1:1.5 risk:reward ratio

    if direction == Side.BUY:
        return entry_price + reward_points
    else:
        return entry_price - reward_points


def _calculate_risk_reward(entry_price: Decimal, sl_price: Decimal, tp_price: Decimal) -> Decimal:
    """Calculate risk:reward ratio."""
    risk = abs(entry_price - sl_price)
    reward = abs(tp_price - entry_price)

    if risk == Decimal('0'):
        return Decimal('0.0')

    return reward / risk


def _calculate_confidence(fvg_quality: float, swing_count: int, sweep_strength: float) -> float:
    """Calculate setup confidence."""
    # Base confidence from FVG quality
    confidence = fvg_quality / 10.0

    # Boost for sweep strength
    confidence += sweep_strength * 0.2

    # Boost for multiple swing confirmations
    if swing_count >= 2:
        confidence += 0.1
    elif swing_count >= 1:
        confidence += 0.05

    return min(1.0, confidence)


def _calculate_fvg_volume_ratio(fvg: FVG, market_state: MarketState) -> float:
    """Calculate volume ratio around FVG creation."""
    if not market_state.bars_5m:
        return 1.0

    # Get volume around FVG creation
    start_idx = max(0, fvg.start_bar - 2)
    end_idx = min(len(market_state.bars_5m), fvg.end_bar + 3)

    if start_idx >= end_idx:
        return 1.0

    fvg_volumes = [market_state.bars_5m[i].volume for i in range(start_idx, end_idx)]
    avg_fvg_volume = np.mean(fvg_volumes) if fvg_volumes else 0

    # Get average volume for comparison
    recent_volumes = [bar.volume for bar in market_state.get_latest_5m_bars(20)]
    avg_recent_volume = np.mean(recent_volumes) if recent_volumes else 1

    if avg_recent_volume == 0:
        return 1.0

    return avg_fvg_volume / avg_recent_volume