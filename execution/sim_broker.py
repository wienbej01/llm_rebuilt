"""
Simulation broker for PSE-LLM trading system.
Handles market/limit fills, slippage/spread models, fees, and SL/TP/partial brackets.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

import numpy as np

from engine.types import (
    Bar,
    ExecutionReport,
    OrderIntent,
    OrderStatus,
    OrderType,
    Side,
    TimeInForce,
)

logger = logging.getLogger(__name__)


class FillType(str, Enum):
    """Fill type enumeration."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


@dataclass
class OrderState:
    """Order state tracking."""
    order_id: str
    symbol: str
    side: Side
    order_type: OrderType
    quantity: int
    filled_quantity: int = 0
    remaining_quantity: int = 0
    price: Decimal | None = None
    stop_price: Decimal | None = None
    time_in_force: TimeInForce = TimeInForce.DAY
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    fills: list[ExecutionReport] = field(default_factory=list)
    average_fill_price: Decimal | None = None
    commission: Decimal = Decimal('0')
    slippage: Decimal = Decimal('0')
    mae: Decimal | None = None  # Maximum Adverse Excursion
    mfe: Decimal | None = None  # Maximum Favorable Excursion
    duration_seconds: float = 0.0
    bracket_orders: list[str] = field(default_factory=list)  # SL/TP order IDs

    def __post_init__(self):
        """Initialize remaining quantity."""
        self.remaining_quantity = self.quantity - self.filled_quantity


@dataclass
class Position:
    """Position tracking."""
    symbol: str
    side: Side
    quantity: int
    average_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    commission: Decimal
    open_time: datetime
    last_update: datetime
    trades_count: int = 0

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.side == Side.BUY

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.side == Side.SELL

    @property
    def is_flat(self) -> bool:
        """Check if position is flat."""
        return self.quantity == 0


@dataclass
class SimBrokerConfig:
    """Simulation broker configuration."""
    initial_capital: Decimal = Decimal('100000.0')
    commission_per_contract: Decimal = Decimal('0.50')
    commission_per_share: Decimal = Decimal('0.005')
    min_commission: Decimal = Decimal('1.00')
    slippage_model: str = "percentage"  # percentage, fixed, volatility
    slippage_percentage: Decimal = Decimal('0.0001')  # 0.01%
    slippage_fixed: Decimal = Decimal('0.25')
    spread_model: str = "dynamic"  # fixed, dynamic, volatility
    spread_percentage: Decimal = Decimal('0.0002')  # 0.02%
    spread_fixed: Decimal = Decimal('0.50')
    fill_probability: float = 0.95  # Probability of limit order fill
    partial_fill_probability: float = 0.3  # Probability of partial fill
    max_partial_fill_ratio: float = 0.5  # Maximum partial fill ratio
    enable_bracket_orders: bool = True
    enable_trailing_stops: bool = True
    trail_amount: Decimal = Decimal('0.50')
    enable_position_sizing: bool = True
    max_position_size: int = 100
    risk_per_trade: Decimal = Decimal('0.01')  # 1% risk per trade
    max_daily_loss: Decimal = Decimal('0.03')  # 3% max daily loss
    enable_statistics: bool = True


class SimBroker:
    """Simulation broker for backtesting and paper trading."""

    def __init__(self, config: SimBrokerConfig | None = None):
        """
        Initialize simulation broker.

        Args:
            config: Broker configuration
        """
        self.config = config or SimBrokerConfig()

        # State tracking
        self.orders: dict[str, OrderState] = {}
        self.positions: dict[str, Position] = {}
        self.executions: list[ExecutionReport] = []
        self.daily_pnl: list[Decimal] = []
        self.equity_curve: list[tuple[datetime, Decimal]] = []

        # Account state
        self.initial_capital = self.config.initial_capital
        self.current_capital = self.initial_capital
        self.daily_loss = Decimal('0')
        self.total_commission = Decimal('0')
        self.total_slippage = Decimal('0')

        # Market data
        self.current_prices: dict[str, Decimal] = {}
        self.current_bars: dict[str, Bar] = {}

        # Statistics
        self.stats = {
            "total_orders": 0,
            "filled_orders": 0,
            "rejected_orders": 0,
            "cancelled_orders": 0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "average_trade_pnl": Decimal('0'),
            "max_drawdown": Decimal('0'),
            "sharpe_ratio": Decimal('0'),
            "profit_factor": Decimal('0')
        }

        # Event loop for async operations
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        logger.info(f"SimBroker initialized with capital: ${self.initial_capital}")

    def update_market_data(self, bar: Bar) -> None:
        """
        Update market data with new bar.

        Args:
            bar: Price bar
        """
        self.current_prices[bar.symbol] = bar.close
        self.current_bars[bar.symbol] = bar

        # Update positions
        self._update_positions(bar.symbol, bar.close)

        # Process orders
        self._process_orders(bar.symbol, bar)

        # Update equity curve
        self._update_equity_curve(bar.timestamp)

    def place_order(self, order_intent: OrderIntent) -> str:
        """
        Place an order.

        Args:
            order_intent: Order intent

        Returns:
            Order ID
        """
        order_id = str(uuid.uuid4())
        order_state = OrderState(
            order_id=order_id,
            symbol=order_intent.symbol,
            side=order_intent.side,
            order_type=order_intent.order_type,
            quantity=order_intent.quantity,
            price=order_intent.price,
            stop_price=order_intent.stop_price,
            time_in_force=order_intent.time_in_force
        )

        self.orders[order_id] = order_state
        self.stats["total_orders"] += 1

        logger.debug(f"Placed order: {order_id} - {order_intent.side} {order_intent.quantity} @ {order_intent.price}")

        # Process immediately if market order
        if order_intent.order_type == OrderType.MARKET:
            self._process_market_order(order_state)

        return order_id

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Order ID

        Returns:
            True if cancelled, False otherwise
        """
        if order_id not in self.orders:
            return False

        order = self.orders[order_id]
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            return False

        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.now(UTC)
        self.stats["cancelled_orders"] += 1

        logger.debug(f"Cancelled order: {order_id}")
        return True

    def place_bracket_order(
        self,
        entry_order: OrderIntent,
        stop_loss: Decimal,
        take_profit: Decimal,
        stop_loss_order_type: OrderType = OrderType.STOP,
        take_profit_order_type: OrderType = OrderType.LIMIT
    ) -> tuple[str, str, str]:
        """
        Place bracket order (entry + SL + TP).

        Args:
            entry_order: Entry order intent
            stop_loss: Stop loss price
            take_profit: Take profit price
            stop_loss_order_type: Stop loss order type
            take_profit_order_type: Take profit order type

        Returns:
            Tuple of (entry_id, stop_loss_id, take_profit_id)
        """
        # Place entry order
        entry_id = self.place_order(entry_order)

        # Create stop loss order
        stop_loss_order = OrderIntent(
            symbol=entry_order.symbol,
            side=Side.SELL if entry_order.side == Side.BUY else Side.BUY,
            order_type=stop_loss_order_type,
            quantity=entry_order.quantity,
            price=stop_loss,
            stop_price=stop_loss if stop_loss_order_type == OrderType.STOP else None,
            time_in_force=TimeInForce.GTC,
            outside_rth=True
        )

        # Create take profit order
        take_profit_order = OrderIntent(
            symbol=entry_order.symbol,
            side=Side.SELL if entry_order.side == Side.BUY else Side.BUY,
            order_type=take_profit_order_type,
            quantity=entry_order.quantity,
            price=take_profit,
            stop_price=None,
            time_in_force=TimeInForce.GTC,
            outside_rth=True
        )

        # Place bracket orders
        stop_loss_id = self.place_order(stop_loss_order)
        take_profit_id = self.place_order(take_profit_order)

        # Link orders
        if entry_id in self.orders:
            self.orders[entry_id].bracket_orders = [stop_loss_id, take_profit_id]

        logger.debug(f"Placed bracket order: entry={entry_id}, sl={stop_loss_id}, tp={take_profit_id}")
        return entry_id, stop_loss_id, take_profit_id

    def get_order_status(self, order_id: str) -> OrderState | None:
        """
        Get order status.

        Args:
            order_id: Order ID

        Returns:
            Order state or None if not found
        """
        return self.orders.get(order_id)

    def get_position(self, symbol: str) -> Position | None:
        """
        Get position for symbol.

        Args:
            symbol: Symbol

        Returns:
            Position or None if no position
        """
        return self.positions.get(symbol)

    def get_all_positions(self) -> dict[str, Position]:
        """
        Get all positions.

        Returns:
            Dictionary of positions by symbol
        """
        return self.positions.copy()

    def get_account_summary(self) -> dict[str, Any]:
        """
        Get account summary.

        Returns:
            Account summary dictionary
        """
        total_position_value = sum(
            pos.quantity * pos.current_price for pos in self.positions.values()
        )

        return {
            "initial_capital": float(self.initial_capital),
            "current_capital": float(self.current_capital),
            "total_position_value": float(total_position_value),
            "total_equity": float(self.current_capital + total_position_value),
            "daily_pnl": float(self.daily_loss),
            "total_commission": float(self.total_commission),
            "total_slippage": float(self.total_slippage),
            "open_orders": len([o for o in self.orders.values() if o.status == OrderStatus.PENDING]),
            "filled_orders": len([o for o in self.orders.values() if o.status == OrderStatus.FILLED]),
            "positions": len(self.positions)
        }

    def get_statistics(self) -> dict[str, Any]:
        """
        Get trading statistics.

        Returns:
            Statistics dictionary
        """
        stats = self.stats.copy()

        # Calculate additional statistics
        if stats["total_trades"] > 0:
            stats["win_rate"] = stats["winning_trades"] / stats["total_trades"]
            stats["loss_rate"] = stats["losing_trades"] / stats["total_trades"]
        else:
            stats["win_rate"] = 0.0
            stats["loss_rate"] = 0.0

        # Calculate profit factor
        if stats["losing_trades"] > 0:
            stats["profit_factor"] = float(
                self.stats["average_trade_pnl"] * stats["winning_trades"] /
                (abs(self.stats["average_trade_pnl"]) * stats["losing_trades"])
            )
        else:
            stats["profit_factor"] = float('inf')

        return stats

    def _process_market_order(self, order: OrderState) -> None:
        """Process market order."""
        if order.symbol not in self.current_prices:
            return

        current_price = self.current_prices[order.symbol]

        # Calculate fill price with slippage
        fill_price = self._calculate_fill_price(current_price, order.side, FillType.MARKET)

        # Calculate fill quantity
        fill_quantity = order.remaining_quantity

        # Execute fill
        self._execute_fill(order, fill_price, fill_quantity)

    def _process_orders(self, symbol: str, bar: Bar) -> None:
        """Process orders for symbol."""
        for order in self.orders.values():
            if order.symbol != symbol or order.status != OrderStatus.PENDING:
                continue

            if order.order_type == OrderType.LIMIT:
                self._process_limit_order(order, bar)
            elif order.order_type == OrderType.STOP:
                self._process_stop_order(order, bar)
            elif order.order_type == OrderType.STOP_LIMIT:
                self._process_stop_limit_order(order, bar)

    def _process_limit_order(self, order: OrderState, bar: Bar) -> None:
        """Process limit order."""
        if order.price is None:
            return

        # Check if order should be filled
        should_fill = False
        if order.side == Side.BUY and bar.low <= order.price:
            should_fill = True
        elif order.side == Side.SELL and bar.high >= order.price:
            should_fill = True

        if should_fill:
            # Check fill probability
            if np.random.random() < self.config.fill_probability:
                # Calculate fill quantity (considering partial fills)
                fill_quantity = self._calculate_fill_quantity(order.remaining_quantity)

                # Calculate fill price
                fill_price = self._calculate_fill_price(order.price, order.side, FillType.LIMIT)

                # Execute fill
                self._execute_fill(order, fill_price, fill_quantity)

    def _process_stop_order(self, order: OrderState, bar: Bar) -> None:
        """Process stop order."""
        if order.stop_price is None:
            return

        # Check if stop is triggered
        stop_triggered = False
        if order.side == Side.BUY and bar.high >= order.stop_price:
            stop_triggered = True
        elif order.side == Side.SELL and bar.low <= order.stop_price:
            stop_triggered = True

        if stop_triggered:
            # Convert to market order
            order.order_type = OrderType.MARKET
            self._process_market_order(order)

    def _process_stop_limit_order(self, order: OrderState, bar: Bar) -> None:
        """Process stop-limit order."""
        if order.stop_price is None or order.price is None:
            return

        # Check if stop is triggered
        stop_triggered = False
        if order.side == Side.BUY and bar.high >= order.stop_price:
            stop_triggered = True
        elif order.side == Side.SELL and bar.low <= order.stop_price:
            stop_triggered = True

        if stop_triggered:
            # Convert to limit order
            order.order_type = OrderType.LIMIT
            self._process_limit_order(order, bar)

    def _calculate_fill_price(self, base_price: Decimal, side: Side, fill_type: FillType) -> Decimal:
        """Calculate fill price with slippage."""
        slippage = self._calculate_slippage(base_price, side, fill_type)
        spread = self._calculate_spread(base_price)

        if side == Side.BUY:
            return base_price + slippage + spread / 2
        else:
            return base_price - slippage - spread / 2

    def _calculate_slippage(self, price: Decimal, side: Side, fill_type: FillType) -> Decimal:
        """Calculate slippage."""
        if self.config.slippage_model == "percentage":
            return price * self.config.slippage_percentage
        elif self.config.slippage_model == "fixed":
            return self.config.slippage_fixed
        else:  # volatility
            # Simple volatility-based slippage (would need historical data for proper implementation)
            return price * self.config.slippage_percentage * 2

    def _calculate_spread(self, price: Decimal) -> Decimal:
        """Calculate spread."""
        if self.config.spread_model == "fixed":
            return self.config.spread_fixed
        elif self.config.spread_model == "dynamic":
            return price * self.config.spread_percentage
        else:  # volatility
            return price * self.config.spread_percentage * 1.5

    def _calculate_fill_quantity(self, remaining_quantity: int) -> int:
        """Calculate fill quantity (considering partial fills)."""
        if np.random.random() < self.config.partial_fill_probability:
            # Partial fill
            fill_ratio = np.random.uniform(0.1, self.config.max_partial_fill_ratio)
            return int(remaining_quantity * fill_ratio)
        else:
            # Full fill
            return remaining_quantity

    def _execute_fill(self, order: OrderState, fill_price: Decimal, fill_quantity: int) -> None:
        """Execute order fill."""
        # Calculate commission
        commission = self._calculate_commission(order.symbol, fill_quantity, fill_price)

        # Calculate slippage
        slippage = abs(fill_price - (order.price or fill_price)) * fill_quantity

        # Create execution report
        execution = ExecutionReport(
            order_id=uuid.UUID(order.order_id),
            symbol=order.symbol,
            side=order.side,
            quantity=fill_quantity,
            price=fill_price,
            commission=commission
        )

        # Update order state
        order.filled_quantity += fill_quantity
        order.remaining_quantity = order.quantity - order.filled_quantity
        order.fills.append(execution)
        order.average_fill_price = self._calculate_average_fill_price(order)
        order.commission += commission
        order.slippage += slippage
        order.updated_at = datetime.now(UTC)

        # Update order status
        if order.remaining_quantity == 0:
            order.status = OrderStatus.FILLED
            order.duration_seconds = (order.updated_at - order.created_at).total_seconds()
            self.stats["filled_orders"] += 1
        else:
            order.status = OrderStatus.PARTIALLY_FILLED

        # Update position
        self._update_position_from_fill(order, execution)

        # Update account
        self._update_account_from_fill(execution)

        # Track execution
        self.executions.append(execution)

        # Cancel bracket orders if filled
        if order.status == OrderStatus.FILLED:
            self._cancel_bracket_orders(order.bracket_orders)

        logger.debug(f"Executed fill: {order.order_id} - {fill_quantity} @ {fill_price}")

    def _calculate_commission(self, symbol: str, quantity: int, price: Decimal) -> Decimal:
        """Calculate commission."""
        # Per contract commission
        contract_commission = quantity * self.config.commission_per_contract

        # Per share commission
        share_commission = quantity * price * self.config.commission_per_share

        # Use maximum of the two with minimum
        commission = max(contract_commission, share_commission)
        commission = max(commission, self.config.min_commission)

        return commission

    def _calculate_average_fill_price(self, order: OrderState) -> Decimal:
        """Calculate average fill price."""
        if not order.fills:
            return Decimal('0')

        total_value = sum(fill.price * fill.quantity for fill in order.fills)
        total_quantity = sum(fill.quantity for fill in order.fills)

        return total_value / total_quantity if total_quantity > 0 else Decimal('0')

    def _update_position_from_fill(self, order: OrderState, execution: ExecutionReport) -> None:
        """Update position from fill."""
        symbol = order.symbol

        if symbol not in self.positions:
            # Create new position
            self.positions[symbol] = Position(
                symbol=symbol,
                side=order.side,
                quantity=execution.quantity,
                average_price=execution.price,
                current_price=execution.price,
                unrealized_pnl=Decimal('0'),
                realized_pnl=Decimal('0'),
                commission=execution.commission,
                open_time=execution.timestamp,
                last_update=execution.timestamp,
                trades_count=1
            )
        else:
            # Update existing position
            position = self.positions[symbol]

            if position.side == order.side:
                # Add to position
                total_quantity = position.quantity + execution.quantity
                total_value = (position.quantity * position.average_price +
                              execution.quantity * execution.price)
                position.average_price = total_value / total_quantity
                position.quantity = total_quantity
            else:
                # Reduce position
                position.quantity -= execution.quantity

                # Calculate realized PnL
                if position.quantity == 0:
                    # Position closed
                    realized_pnl = (execution.price - position.average_price) * execution.quantity
                    if position.side == Side.SELL:
                        realized_pnl = -realized_pnl
                    position.realized_pnl += realized_pnl
                    self.stats["total_trades"] += 1

                    if realized_pnl > 0:
                        self.stats["winning_trades"] += 1
                    else:
                        self.stats["losing_trades"] += 1

                    position.trades_count += 1

            position.last_update = execution.timestamp
            position.commission += execution.commission

    def _update_positions(self, symbol: str, current_price: Decimal) -> None:
        """Update positions with current price."""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        position.current_price = current_price

        # Calculate unrealized PnL
        if position.quantity > 0:
            if position.side == Side.BUY:
                position.unrealized_pnl = (current_price - position.average_price) * position.quantity
            else:
                position.unrealized_pnl = (position.average_price - current_price) * position.quantity

        # Update MAE/MFE
        if position.unrealized_pnl > (position.mfe or Decimal('0')):
            position.mfe = position.unrealized_pnl
        if position.unrealized_pnl < (position.mae or Decimal('0')):
            position.mae = position.unrealized_pnl

    def _update_account_from_fill(self, execution: ExecutionReport) -> None:
        """Update account from fill."""
        # Update commission
        self.total_commission += execution.commission

        # Update daily PnL (would be calculated from position changes)
        # This is simplified - in reality would track daily PnL more carefully
        pass

    def _update_equity_curve(self, timestamp: datetime) -> None:
        """Update equity curve."""
        total_equity = self.current_capital + sum(
            pos.quantity * pos.current_price for pos in self.positions.values()
        )
        self.equity_curve.append((timestamp, total_equity))

    def _cancel_bracket_orders(self, bracket_order_ids: list[str]) -> None:
        """Cancel bracket orders."""
        for order_id in bracket_order_ids:
            self.cancel_order(order_id)

    def reset(self) -> None:
        """Reset broker state."""
        self.orders.clear()
        self.positions.clear()
        self.executions.clear()
        self.daily_pnl.clear()
        self.equity_curve.clear()

        self.current_capital = self.initial_capital
        self.daily_loss = Decimal('0')
        self.total_commission = Decimal('0')
        self.total_slippage = Decimal('0')

        self.stats = {
            "total_orders": 0,
            "filled_orders": 0,
            "rejected_orders": 0,
            "cancelled_orders": 0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "average_trade_pnl": Decimal('0'),
            "max_drawdown": Decimal('0'),
            "sharpe_ratio": Decimal('0'),
            "profit_factor": Decimal('0')
        }

        logger.info("SimBroker reset")

    def get_state(self) -> dict[str, Any]:
        """Get broker state for serialization."""
        return {
            "config": {
                "initial_capital": float(self.config.initial_capital),
                "commission_per_contract": float(self.config.commission_per_contract),
                "commission_per_share": float(self.config.commission_per_share),
                "slippage_percentage": float(self.config.slippage_percentage),
                "spread_percentage": float(self.config.spread_percentage)
            },
            "account": self.get_account_summary(),
            "statistics": self.get_statistics(),
            "orders": {oid: self._order_to_dict(order) for oid, order in self.orders.items()},
            "positions": {symbol: self._position_to_dict(pos) for symbol, pos in self.positions.items()},
            "equity_curve": [(ts.isoformat(), float(eq)) for ts, eq in self.equity_curve]
        }

    def _order_to_dict(self, order: OrderState) -> dict[str, Any]:
        """Convert order to dictionary."""
        return {
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "order_type": order.order_type.value,
            "quantity": order.quantity,
            "filled_quantity": order.filled_quantity,
            "remaining_quantity": order.remaining_quantity,
            "price": float(order.price) if order.price else None,
            "stop_price": float(order.stop_price) if order.stop_price else None,
            "status": order.status.value,
            "created_at": order.created_at.isoformat(),
            "updated_at": order.updated_at.isoformat(),
            "average_fill_price": float(order.average_fill_price) if order.average_fill_price else None,
            "commission": float(order.commission),
            "slippage": float(order.slippage),
            "duration_seconds": order.duration_seconds
        }

    def _position_to_dict(self, position: Position) -> dict[str, Any]:
        """Convert position to dictionary."""
        return {
            "symbol": position.symbol,
            "side": position.side.value,
            "quantity": position.quantity,
            "average_price": float(position.average_price),
            "current_price": float(position.current_price),
            "unrealized_pnl": float(position.unrealized_pnl),
            "realized_pnl": float(position.realized_pnl),
            "commission": float(position.commission),
            "open_time": position.open_time.isoformat(),
            "last_update": position.last_update.isoformat(),
            "trades_count": position.trades_count,
            "mae": float(position.mae) if position.mae else None,
            "mfe": float(position.mfe) if position.mfe else None
        }

    def __del__(self):
        """Cleanup event loop."""
        if hasattr(self, 'loop') and self.loop.is_running():
            self.loop.close()
