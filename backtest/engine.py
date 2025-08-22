"""
Core backtesting engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from decimal import Decimal

from engine.strategy_engine import StrategyEngine, EngineConfig
from engine.types import Bar, OrderIntent, OrderType, Side, SetupProposal
from execution.sim_broker import SimBroker, SimBrokerConfig
from backtest.metrics import calculate_metrics

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for the backtest."""
    symbol: str
    start_date: datetime
    end_date: datetime
    symbol_info: Dict[str, Any]
    initial_capital: Decimal = Decimal("100000.0")


@dataclass
class BacktestResult:
    """Results of a backtest run."""
    config: BacktestConfig
    metrics: Dict[str, Any] = field(default_factory=dict)
    engine_stats: Dict[str, Any] = field(default_factory=dict)
    broker_state: Dict[str, Any] = field(default_factory=dict)
    setups_generated: int = 0
    orders_placed: int = 0


class BacktestEngine:
    """Orchestrates the backtesting process."""

    def __init__(
        self,
        engine_config: EngineConfig,
        broker_config: SimBrokerConfig,
        backtest_config: BacktestConfig
    ):
        self.engine_config = engine_config
        self.broker_config = broker_config
        self.backtest_config = backtest_config

        self.strategy_engine = StrategyEngine(config=self.engine_config)
        self.broker = SimBroker(config=self.broker_config)
        self.broker.current_capital = self.backtest_config.initial_capital
        self.broker.initial_capital = self.backtest_config.initial_capital

    def run(self, bars_5m: List[Bar], bars_1m: List[Bar]) -> BacktestResult:
        """
        Run the backtest.

        Args:
            bars_5m: List of 5-minute bars.
            bars_1m: List of 1-minute bars.

        Returns:
            BacktestResult object.
        """
        logger.info(f"Starting backtest for {self.backtest_config.symbol}...")

        bars_1m_by_5m_ts = {bar.timestamp: [] for bar in bars_5m}
        for bar_1m in bars_1m:
            ts_5m = bar_1m.timestamp.replace(minute=bar_1m.timestamp.minute - bar_1m.timestamp.minute % 5, second=0, microsecond=0)
            if ts_5m in bars_1m_by_5m_ts:
                bars_1m_by_5m_ts[ts_5m].append(bar_1m)

        for i, bar_5m in enumerate(bars_5m):
            bars_1m_context = bars_1m_by_5m_ts.get(bar_5m.timestamp, [])
            if not bars_1m_context:
                continue

            proposals = self.strategy_engine.process_5m_bar_with_1m_context(
                bar_5m, bars_1m_context, self.backtest_config.symbol_info
            )

            for setup in proposals:
                if setup.risk_assessment and setup.risk_assessment.get("is_viable"):
                    pos_size = setup.risk_assessment.get("position_size", 0)
                    if pos_size > 0:
                        entry_order = OrderIntent(
                            symbol=setup.symbol, side=setup.side, order_type=OrderType.MARKET,
                            quantity=pos_size, setup_id=setup.id
                        )
                        self.broker.place_bracket_order(
                            entry_order=entry_order,
                            stop_loss=Decimal(str(setup.stop_loss)),
                            take_profit=Decimal(str(setup.take_profit))
                        )

            for bar_1m in bars_1m_context:
                self.broker.update_market_data(bar_1m)

        logger.info("Backtest finished. Calculating metrics...")
        final_positions = list(self.broker.get_all_positions().values())
        metrics = calculate_metrics(final_positions, self.broker.equity_curve)

        return BacktestResult(
            config=self.backtest_config,
            metrics=metrics,
            engine_stats=self.strategy_engine.get_processing_stats(),
            broker_state=self.broker.get_state(),
            setups_generated=self.strategy_engine.processing_stats["setups_generated"],
            orders_placed=self.broker.stats["total_orders"]
        )
