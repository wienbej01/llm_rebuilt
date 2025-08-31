"""
Risk kernel for PSE-LLM trading system.
Implements universal risk management, position sizing, and exposure controls.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

from numba import float64, jit

from engine.state import MarketState
from engine.trading_types import ExecutionReport, OrderIntent, SetupProposal

logger = logging.getLogger(__name__)


class RiskLimitType(str, Enum):
    """Risk limit type enumeration."""
    STOP_LOSS_CAP = "stop_loss_cap"
    MAX_DAILY_LOSS = "max_daily_loss"
    MAX_CONCURRENT_RISK = "max_concurrent_risk"
    MAX_TRADES_PER_DAY = "max_trades_per_day"
    COOL_OFF_PERIOD = "cool_off_period"
    SPREAD_GUARD = "spread_guard"
    LATENCY_GUARD = "latency_guard"


class RiskKernel:
    """Kernel for risk management and position sizing."""

    def __init__(
        self,
        universal_sl_cap: float = 10.0,
        max_daily_loss_pct: float = 0.03,
        max_concurrent_risk_pct: float = 0.05,
        max_trades_per_day: int = 6,
        cool_off_bars: int = 2,
        max_spread_pct: float = 0.001,
        max_latency_ms: int = 100
    ):
        """
        Initialize risk kernel.

        Args:
            universal_sl_cap: Universal stop loss cap in points
            max_daily_loss_pct: Maximum daily loss as percentage of equity
            max_concurrent_risk_pct: Maximum concurrent risk as percentage of equity
            max_trades_per_day: Maximum trades per day
            cool_off_bars: Number of bars to cool off after a loss
            max_spread_pct: Maximum spread percentage
            max_latency_ms: Maximum latency in milliseconds
        """
        self.universal_sl_cap = universal_sl_cap
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_concurrent_risk_pct = max_concurrent_risk_pct
        self.max_trades_per_day = max_trades_per_day
        self.cool_off_bars = cool_off_bars
        self.max_spread_pct = max_spread_pct
        self.max_latency_ms = max_latency_ms

        # Risk state tracking
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_trade_bar = 0
        self.concurrent_risk = 0.0
        self.last_loss_bar = 0
        self.equity = 100000.0  # Starting equity

    def assess_risk_for_setup(
        self,
        setup: SetupProposal,
        market_state: MarketState,
        symbol_info: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Assess risk for a setup proposal.

        Args:
            setup: Setup proposal to assess
            market_state: Current market state
            symbol_info: Symbol information (tick size, point value, etc.)

        Returns:
            Dictionary with risk assessment results
        """
        risk_assessment = {
            "is_viable": True,
            "risk_score": 0.0,
            "risk_limit_checks": {},
            "position_size": 0,
            "adjusted_sl": None,
            "adjusted_tp1": None,
            "risk_amount": 0.0,
            "reasoning": []
        }

        # Check all risk limits
        limit_checks = self._check_all_risk_limits(setup, market_state, symbol_info)
        risk_assessment["risk_limit_checks"] = limit_checks

        # If any limit is violated, mark as not viable
        if any(check["violated"] for check in limit_checks.values()):
            risk_assessment["is_viable"] = False
            risk_assessment["reasoning"].append("Risk limits violated")

        # Calculate position size if viable
        if risk_assessment["is_viable"]:
            position_size = self._calculate_position_size(setup, symbol_info)
            risk_assessment["position_size"] = position_size

            # Calculate risk amount
            risk_amount = self._calculate_risk_amount(setup, position_size, symbol_info)
            risk_assessment["risk_amount"] = risk_amount

            # Adjust SL and TP if needed
            adjusted_sl, adjusted_tp1 = self._adjust_sl_tp(setup, symbol_info)
            risk_assessment["adjusted_sl"] = adjusted_sl
            risk_assessment["adjusted_tp1"] = adjusted_tp1

            # Calculate risk score
            risk_score = self._calculate_risk_score(setup, market_state, risk_amount)
            risk_assessment["risk_score"] = risk_score

        return risk_assessment

    def check_order_intent_risk(
        self,
        order_intent: OrderIntent,
        market_state: MarketState,
        symbol_info: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Check risk for an order intent before execution.

        Args:
            order_intent: Order intent to check
            market_state: Current market state
            symbol_info: Symbol information

        Returns:
            Dictionary with order risk assessment
        """
        risk_check = {
            "can_execute": True,
            "risk_violations": [],
            "adjusted_quantity": order_intent.quantity,
            "reasoning": []
        }

        # Check concurrent risk
        concurrent_risk_check = self._check_concurrent_risk(order_intent, symbol_info)
        if not concurrent_risk_check["passed"]:
            risk_check["can_execute"] = False
            risk_check["risk_violations"].append("concurrent_risk")
            risk_check["reasoning"].append(concurrent_risk_check["reason"])

        # Check daily limits
        daily_check = self._check_daily_limits(order_intent, symbol_info)
        if not daily_check["passed"]:
            risk_check["can_execute"] = False
            risk_check["risk_violations"].append("daily_limits")
            risk_check["reasoning"].append(daily_check["reason"])

        # Check spread and latency
        market_conditions_check = self._check_market_conditions(order_intent, market_state)
        if not market_conditions_check["passed"]:
            risk_check["can_execute"] = False
            risk_check["risk_violations"].append("market_conditions")
            risk_check["reasoning"].append(market_conditions_check["reason"])

        # Adjust quantity if needed
        if risk_check["can_execute"]:
            adjusted_quantity = self._adjust_quantity_for_risk(order_intent, symbol_info)
            risk_check["adjusted_quantity"] = adjusted_quantity

        return risk_check

    def update_risk_state(self, execution_report: ExecutionReport, market_state: MarketState) -> None:
        """
        Update risk state based on execution report.

        Args:
            execution_report: Execution report to process
            market_state: Current market state
        """
        # Update daily P&L
        if execution_report.status == "FILLED":
            # This is a simplified P&L calculation
            # In practice, you'd need to track entry prices and calculate actual P&L
            pnl_change = 0.0  # Placeholder
            self.daily_pnl += pnl_change

        # Update trade count
        self.daily_trades += 1

        # Update last trade bar
        if market_state.bars_5m:
            self.last_trade_bar = len(market_state.bars_5m) - 1

        # Update concurrent risk (simplified)
        # In practice, you'd track open positions and their risk
        self.concurrent_risk = 0.0  # Placeholder

        # Update loss tracking
        if execution_report.status == "FILLED" and self.daily_pnl < 0:
            self.last_loss_bar = len(market_state.bars_5m) - 1 if market_state.bars_5m else 0

        logger.debug(f"Updated risk state: Daily P&L={self.daily_pnl}, Trades={self.daily_trades}")

    def _check_all_risk_limits(
        self,
        setup: SetupProposal,
        market_state: MarketState,
        symbol_info: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        """Check all risk limits for a setup."""
        checks = {}

        # Stop loss cap
        checks[RiskLimitType.STOP_LOSS_CAP] = self._check_sl_cap(setup, symbol_info)

        # Daily loss limit
        checks[RiskLimitType.MAX_DAILY_LOSS] = self._check_daily_loss_limit()

        # Concurrent risk
        checks[RiskLimitType.MAX_CONCURRENT_RISK] = self._check_concurrent_risk_limit(setup, symbol_info)

        # Trades per day
        checks[RiskLimitType.MAX_TRADES_PER_DAY] = self._check_trades_per_day()

        # Cool off period
        checks[RiskLimitType.COOL_OFF_PERIOD] = self._check_cool_off_period(market_state)

        # Spread guard
        checks[RiskLimitType.SPREAD_GUARD] = self._check_spread_guard(market_state, symbol_info)

        # Latency guard
        checks[RiskLimitType.LATENCY_GUARD] = self._check_latency_guard()

        return checks

    def _check_sl_cap(self, setup: SetupProposal, symbol_info: dict[str, Any]) -> dict[str, Any]:
        """Check stop loss cap."""
        sl_points = abs(setup.stop_loss - setup.entry_price)
        tick_size = symbol_info.get("tick_size", 0.25)

        check = {
            "limit": self.universal_sl_cap,
            "actual": sl_points,
            "violated": False,
            "reason": ""
        }

        if sl_points > self.universal_sl_cap:
            check["violated"] = True
            check["reason"] = f"SL {sl_points} exceeds cap {self.universal_sl_cap}"

        return check

    def _check_daily_loss_limit(self) -> dict[str, Any]:
        """Check daily loss limit."""
        daily_loss_limit = self.equity * self.max_daily_loss_pct

        check = {
            "limit": daily_loss_limit,
            "actual": abs(self.daily_pnl),
            "violated": False,
            "reason": ""
        }

        if self.daily_pnl < -daily_loss_limit:
            check["violated"] = True
            check["reason"] = f"Daily loss {abs(self.daily_pnl)} exceeds limit {daily_loss_limit}"

        return check

    def _check_concurrent_risk_limit(
        self,
        setup: SetupProposal,
        symbol_info: dict[str, Any]
    ) -> dict[str, Any]:
        """Check concurrent risk limit."""
        # Calculate risk for this setup
        position_size = self._calculate_position_size(setup, symbol_info)
        risk_amount = self._calculate_risk_amount(setup, position_size, symbol_info)

        concurrent_risk_limit = self.equity * self.max_concurrent_risk_pct
        projected_risk = self.concurrent_risk + risk_amount

        check = {
            "limit": concurrent_risk_limit,
            "actual": projected_risk,
            "violated": False,
            "reason": ""
        }

        if projected_risk > concurrent_risk_limit:
            check["violated"] = True
            check["reason"] = f"Projected risk {projected_risk} exceeds limit {concurrent_risk_limit}"

        return check

    def _check_trades_per_day(self) -> dict[str, Any]:
        """Check trades per day limit."""
        check = {
            "limit": self.max_trades_per_day,
            "actual": self.daily_trades,
            "violated": False,
            "reason": ""
        }

        if self.daily_trades >= self.max_trades_per_day:
            check["violated"] = True
            check["reason"] = f"Daily trades {self.daily_trades} exceeds limit {self.max_trades_per_day}"

        return check

    def _check_cool_off_period(self, market_state: MarketState) -> dict[str, Any]:
        """Check cool off period after loss."""
        if not market_state.bars_5m:
            return {
                "limit": self.cool_off_bars,
                "actual": 0,
                "violated": False,
                "reason": ""
            }

        current_bar = len(market_state.bars_5m) - 1
        bars_since_loss = current_bar - self.last_loss_bar

        check = {
            "limit": self.cool_off_bars,
            "actual": bars_since_loss,
            "violated": False,
            "reason": ""
        }

        if bars_since_loss < self.cool_off_bars and self.daily_pnl < 0:
            check["violated"] = True
            check["reason"] = f"Cool off period: {bars_since_loss} < {self.cool_off_bars} bars"

        return check

    def _check_spread_guard(
        self,
        market_state: MarketState,
        symbol_info: dict[str, Any]
    ) -> dict[str, Any]:
        """Check spread guard."""
        # This is a simplified implementation
        # In practice, you'd get real-time spread data
        estimated_spread = symbol_info.get("estimated_spread", 0.25)
        avg_price = symbol_info.get("avg_price", 5000.0)

        spread_pct = estimated_spread / avg_price

        check = {
            "limit": self.max_spread_pct,
            "actual": spread_pct,
            "violated": False,
            "reason": ""
        }

        if spread_pct > self.max_spread_pct:
            check["violated"] = True
            check["reason"] = f"Spread {spread_pct} exceeds limit {self.max_spread_pct}"

        return check

    def _check_latency_guard(self) -> dict[str, Any]:
        """Check latency guard."""
        # This is a simplified implementation
        # In practice, you'd measure actual latency
        estimated_latency = 50  # ms

        check = {
            "limit": self.max_latency_ms,
            "actual": estimated_latency,
            "violated": False,
            "reason": ""
        }

        if estimated_latency > self.max_latency_ms:
            check["violated"] = True
            check["reason"] = f"Latency {estimated_latency}ms exceeds limit {self.max_latency_ms}ms"

        return check

    def _calculate_position_size(self, setup: SetupProposal, symbol_info: dict[str, Any]) -> int:
        """Calculate position size based on risk."""
        # Risk per trade (1R)
        risk_per_trade = self.equity * 0.01  # 1% of equity

        # Calculate risk per contract
        sl_points = abs(setup.stop_loss - setup.entry_price)
        point_value = symbol_info.get("point_value", 50.0)
        risk_per_contract = sl_points * point_value

        if risk_per_contract == 0:
            return 0

        # Calculate position size
        position_size = int(risk_per_trade / risk_per_contract)

        return max(0, position_size)

    def _calculate_risk_amount(
        self,
        setup: SetupProposal,
        position_size: int,
        symbol_info: dict[str, Any]
    ) -> float:
        """Calculate risk amount for a position."""
        sl_points = abs(setup.stop_loss - setup.entry_price)
        point_value = symbol_info.get("point_value", 50.0)

        return sl_points * point_value * position_size

    def _adjust_sl_tp(
        self,
        setup: SetupProposal,
        symbol_info: dict[str, Any]
    ) -> tuple[float, float]:
        """Adjust stop loss and take profit if needed."""
        adjusted_sl = setup.stop_loss
        adjusted_tp1 = setup.take_profit

        # Ensure SL doesn't exceed universal cap
        sl_points = abs(setup.stop_loss - setup.entry_price)
        if sl_points > self.universal_sl_cap:
            if setup.side == "BUY":
                adjusted_sl = setup.entry_price - self.universal_sl_cap
            else:
                adjusted_sl = setup.entry_price + self.universal_sl_cap

        # Ensure TP is at least 1.5x SL away
        new_sl_points = abs(adjusted_sl - setup.entry_price)
        min_tp_points = new_sl_points * 1.5

        tp_points = abs(setup.take_profit - setup.entry_price)
        if tp_points < min_tp_points:
            if setup.side == "BUY":
                adjusted_tp1 = setup.entry_price + min_tp_points
            else:
                adjusted_tp1 = setup.entry_price - min_tp_points

        return adjusted_sl, adjusted_tp1

    def _calculate_risk_score(
        self,
        setup: SetupProposal,
        market_state: MarketState,
        risk_amount: float
    ) -> float:
        """Calculate risk score (0-10, lower is better)."""
        score = 0.0

        # Base score on risk amount
        risk_pct = risk_amount / self.equity
        if risk_pct > 0.02:  # More than 2% risk
            score += 5.0
        elif risk_pct > 0.015:  # 1.5-2% risk
            score += 3.0
        elif risk_pct > 0.01:  # 1-1.5% risk
            score += 1.0

        # Adjust for setup quality
        if hasattr(setup, 'confidence'):
            score += (1.0 - setup.confidence) * 3.0

        # Adjust for market conditions
        if market_state.current_mcs and market_state.current_mcs.bias == "Range":
            score += 2.0  # Higher risk in range-bound markets

        return min(10.0, score)

    def _check_concurrent_risk(
        self,
        order_intent: OrderIntent,
        symbol_info: dict[str, Any]
    ) -> dict[str, Any]:
        """Check concurrent risk for order intent."""
        # Simplified implementation
        concurrent_risk_limit = self.equity * self.max_concurrent_risk_pct

        check = {
            "passed": True,
            "reason": ""
        }

        if self.concurrent_risk > concurrent_risk_limit:
            check["passed"] = False
            check["reason"] = f"Concurrent risk {self.concurrent_risk} exceeds limit {concurrent_risk_limit}"

        return check

    def _check_daily_limits(
        self,
        order_intent: OrderIntent,
        symbol_info: dict[str, Any]
    ) -> dict[str, Any]:
        """Check daily limits for order intent."""
        daily_loss_limit = self.equity * self.max_daily_loss_pct

        check = {
            "passed": True,
            "reason": ""
        }

        if self.daily_pnl < -daily_loss_limit:
            check["passed"] = False
            check["reason"] = f"Daily loss {abs(self.daily_pnl)} exceeds limit {daily_loss_limit}"

        if self.daily_trades >= self.max_trades_per_day:
            check["passed"] = False
            check["reason"] = f"Daily trades {self.daily_trades} exceeds limit {self.max_trades_per_day}"

        return check

    def _check_market_conditions(
        self,
        order_intent: OrderIntent,
        market_state: MarketState
    ) -> dict[str, Any]:
        """Check market conditions for order intent."""
        check = {
            "passed": True,
            "reason": ""
        }

        # Simplified implementation
        # In practice, you'd check real-time spreads and latency
        estimated_spread = 0.25
        estimated_latency = 50  # ms

        if estimated_spread > self.max_spread_pct * 5000:  # Assuming $5000 avg price
            check["passed"] = False
            check["reason"] = f"Spread too wide: {estimated_spread}"

        if estimated_latency > self.max_latency_ms:
            check["passed"] = False
            check["reason"] = f"Latency too high: {estimated_latency}ms"

        return check

    def _adjust_quantity_for_risk(
        self,
        order_intent: OrderIntent,
        symbol_info: dict[str, Any]
    ) -> int:
        """Adjust order quantity for risk management."""
        # Simplified implementation
        # In practice, you'd calculate based on available risk capacity
        return order_intent.quantity

    def get_risk_summary(self) -> dict[str, Any]:
        """Get current risk summary."""
        return {
            "daily_pnl": self.daily_pnl,
            "daily_trades": self.daily_trades,
            "concurrent_risk": self.concurrent_risk,
            "last_trade_bar": self.last_trade_bar,
            "last_loss_bar": self.last_loss_bar,
            "equity": self.equity,
            "risk_limits": {
                "universal_sl_cap": self.universal_sl_cap,
                "max_daily_loss_pct": self.max_daily_loss_pct,
                "max_concurrent_risk_pct": self.max_concurrent_risk_pct,
                "max_trades_per_day": self.max_trades_per_day,
                "cool_off_bars": self.cool_off_bars,
                "max_spread_pct": self.max_spread_pct,
                "max_latency_ms": self.max_latency_ms
            }
        }

    def reset_daily_state(self) -> None:
        """Reset daily risk state."""
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_trade_bar = 0
        self.last_loss_bar = 0
        logger.info("Daily risk state reset")


# Numba-optimized functions
@jit(float64(float64, float64, float64), nopython=True)
def calculate_position_size_numba(equity: float64, risk_pct: float64, risk_per_contract: float64) -> float64:
    """Numba-optimized position size calculation."""
    if risk_per_contract == 0:
        return 0.0

    risk_amount = equity * risk_pct
    return risk_amount / risk_per_contract


@jit(float64(float64, float64, float64), nopython=True)
def calculate_risk_amount_numba(sl_points: float64, point_value: float64, quantity: float64) -> float64:
    """Numba-optimized risk amount calculation."""
    return sl_points * point_value * quantity
