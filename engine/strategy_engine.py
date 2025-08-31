"""
Strategy Engine for PSE-LLM trading system.
Main engine that processes market data and coordinates all components.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from engine.detectors.registry import detector_registry
from engine.kernels.micro_1m import Micro1mKernel
from engine.kernels.quality import QualityKernel
from engine.kernels.risk import RiskKernel
from engine.kernels.structure import StructureKernel
from engine.kernels.tcc_mcs import TCCMCSKernel
from engine.state import MarketState
from engine.trading_types import Bar, SetupProposal, Side

logger = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    """Configuration for the strategy engine."""
    enable_hooks: bool = True
    enable_state_snapshot: bool = True
    max_bars_5m: int = 1000
    max_bars_1m: int = 5000
    max_swings: int = 100
    max_mss: int = 50
    max_fvgs: int = 100
    enable_detectors: list[str] = None

    def __post_init__(self):
        if self.enable_detectors is None:
            self.enable_detectors = ["das21", "das22_sweep", "tcea_fvg_mr", "tcea_spmf_v3", "mte"]


class StrategyEngine:
    """Main strategy engine for PSE-LLM trading system."""

    def __init__(self, config: EngineConfig | None = None):
        """
        Initialize strategy engine.

        Args:
            config: Engine configuration
        """
        self.config = config or EngineConfig()
        self.market_state = MarketState()

        # Initialize kernels
        self.structure_kernel = StructureKernel()
        self.quality_kernel = QualityKernel()
        self.tcc_mcs_kernel = TCCMCSKernel()
        self.micro_1m_kernel = Micro1mKernel()
        self.risk_kernel = RiskKernel()

        # Hook system
        self.hooks: dict[str, list[Callable]] = {
            "pre_analysis": [],
            "post_detection": [],
            "pre_llm": [],
            "post_llm": []
        }

        # Performance tracking
        self.processing_stats = {
            "bars_processed": 0,
            "setups_generated": 0,
            "processing_time_ms": 0,
            "last_processed": None
        }

        logger.info("Strategy engine initialized")

    def register_hook(self, hook_name: str, hook_func: Callable) -> None:
        """
        Register a hook function.

        Args:
            hook_name: Name of the hook
            hook_func: Hook function to register
        """
        if hook_name in self.hooks:
            self.hooks[hook_name].append(hook_func)
            logger.info(f"Registered hook: {hook_name}")
        else:
            logger.warning(f"Unknown hook name: {hook_name}")

    def unregister_hook(self, hook_name: str, hook_func: Callable) -> None:
        """
        Unregister a hook function.

        Args:
            hook_name: Name of the hook
            hook_func: Hook function to unregister
        """
        if hook_name in self.hooks and hook_func in self.hooks[hook_name]:
            self.hooks[hook_name].remove(hook_func)
            logger.info(f"Unregistered hook: {hook_name}")

    def _run_hooks(self, hook_name: str, *args, **kwargs) -> None:
        """Run all hooks for a given hook name."""
        if not self.config.enable_hooks:
            return

        for hook_func in self.hooks.get(hook_name, []):
            try:
                hook_func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in hook {hook_name}: {e}")

    def process_5m_bar_with_1m_context(
        self,
        bar_5m: Bar,
        bars_1m_window: list[Bar]
    ) -> list[SetupProposal]:
        """
        Process a 5-minute bar with 1-minute context and return setup proposals.

        Args:
            bar_5m: 5-minute bar to process
            bars_1m_window: 1-minute bars for context

        Returns:
            List of setup proposals
        """
        start_time = datetime.now(UTC)

        try:
            # Run pre-analysis hooks
            self._run_hooks("pre_analysis", bar_5m, bars_1m_window)

            # Update market state
            self._update_market_state(bar_5m, bars_1m_window)

            # Run structure kernel
            structure_update = self.structure_kernel.update_structure(
                self.market_state, bar_5m, bars_1m_window
            )

            # Run quality kernel
            quality_update = self.quality_kernel.update_quality(
                self.market_state, structure_update
            )

            # Run TCC/MCS kernel
            tcc_mcs_update = self.tcc_mcs_kernel.update_tcc_mcs(
                self.market_state, bar_5m
            )

            # Run micro 1m kernel
            micro_analysis = self.micro_1m_kernel.analyze_micro_structure(
                self.market_state, bars_1m_window
            )

            # Run detectors
            setups = self._run_detectors(bars_1m_window)

            # Run post-detection hooks
            self._run_hooks("post_detection", setups)

            # Run pre-LLM hooks
            self._run_hooks("pre_llm", setups)

            # TODO: LLM integration would go here
            # llm_validated_setups = self._validate_with_llm(setups)

            # Run post-LLM hooks
            self._run_hooks("post_llm", setups)

            # Update processing stats
            self._update_processing_stats(start_time, len(setups))

            logger.debug(f"Processed 5m bar {bar_5m.timestamp}, generated {len(setups)} setups")

            return setups

        except Exception as e:
            logger.error(f"Error processing 5m bar: {e}")
            return []

    def _update_market_state(self, bar_5m: Bar, bars_1m_window: list[Bar]) -> None:
        """Update market state with new bars."""
        # Add 5m bar
        self.market_state.add_5m_bar(bar_5m)

        # Add 1m bars
        for bar_1m in bars_1m_window:
            self.market_state.add_1m_bar(bar_1m)

        # Prune old data if needed
        self._prune_market_state()

    def _prune_market_state(self) -> None:
        """Prune old market state data to manage memory."""
        # Prune 5m bars
        if len(self.market_state.bars_5m) > self.config.max_bars_5m:
            self.market_state.bars_5m = self.market_state.bars_5m[-self.config.max_bars_5m:]

        # Prune 1m bars
        if len(self.market_state.bars_1m) > self.config.max_bars_1m:
            self.market_state.bars_1m = self.market_state.bars_1m[-self.config.max_bars_1m:]

        # Prune swings
        if len(self.market_state.swings) > self.config.max_swings:
            self.market_state.swings = self.market_state.swings[-self.config.max_swings:]

        # Prune MSS
        if len(self.market_state.mss_list) > self.config.max_mss:
            self.market_state.mss_list = self.market_state.mss_list[-self.config.max_mss:]

        # Prune FVGs
        if len(self.market_state.fvgs) > self.config.max_fvgs:
            self.market_state.fvgs = self.market_state.fvgs[-self.config.max_fvgs:]

    def _run_detectors(self, bars_1m_window: list[Bar]) -> list[SetupProposal]:
        """Run enabled detectors and return setup proposals."""
        setups = []

        # Enable configured detectors
        for detector_name in self.config.enable_detectors:
            detector_registry.enable_detector(detector_name)

        # Run all enabled detectors
        setups = detector_registry.run_all_detectors(self.market_state, bars_1m_window)

        # Assess risk for each setup
        risk_assessed_setups = []
        for setup in setups:
            # Get symbol info (simplified)
            symbol_info = {
                "tick_size": 0.25,
                "point_value": 50.0,
                "estimated_spread": 0.25,
                "avg_price": 5000.0
            }

            risk_assessment = self.risk_kernel.assess_risk_for_setup(
                setup, self.market_state, symbol_info
            )

            if risk_assessment["is_viable"]:
                # Update setup with risk assessment
                setup.risk_assessment = risk_assessment
                risk_assessed_setups.append(setup)

        return risk_assessed_setups

    def _update_processing_stats(self, start_time: datetime, setup_count: int) -> None:
        """Update processing statistics."""
        processing_time = (datetime.now(UTC) - start_time).total_seconds() * 1000

        self.processing_stats["bars_processed"] += 1
        self.processing_stats["setups_generated"] += setup_count
        self.processing_stats["processing_time_ms"] += processing_time
        self.processing_stats["last_processed"] = datetime.now(UTC)

    def get_processing_stats(self) -> dict[str, Any]:
        """Get processing statistics."""
        stats = self.processing_stats.copy()

        # Calculate averages
        if stats["bars_processed"] > 0:
            stats["avg_processing_time_ms"] = stats["processing_time_ms"] / stats["bars_processed"]
            stats["setups_per_bar"] = stats["setups_generated"] / stats["bars_processed"]

        return stats

    def get_market_state_snapshot(self) -> dict[str, Any]:
        """Get current market state snapshot."""
        if not self.config.enable_state_snapshot:
            return {}

        return {
            "bars_5m_count": len(self.market_state.bars_5m),
            "bars_1m_count": len(self.market_state.bars_1m),
            "swings_count": len(self.market_state.swings),
            "mss_count": len(self.market_state.mss_list),
            "fvgs_count": len(self.market_state.fvgs),
            "current_tcc": asdict(self.market_state.current_tcc) if self.market_state.current_tcc else None,
            "current_mcs": asdict(self.market_state.current_mcs) if self.market_state.current_mcs else None,
            "last_update": self.market_state.last_update.isoformat() if self.market_state.last_update else None
        }

    def save_state(self, filepath: str) -> None:
        """Save engine state to file."""
        state = {
            "config": asdict(self.config),
            "market_state": self.market_state.to_dict(),
            "processing_stats": self.processing_stats,
            "detector_configs": detector_registry.export_config_to_dict()
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"Saved engine state to {filepath}")

    def load_state(self, filepath: str) -> None:
        """Load engine state from file."""
        with open(filepath) as f:
            state = json.load(f)

        # Load config
        self.config = EngineConfig(**state["config"])

        # Load market state
        self.market_state.from_dict(state["market_state"])

        # Load processing stats
        self.processing_stats = state["processing_stats"]

        # Load detector configs
        detector_registry.load_config_from_dict(state["detector_configs"])

        logger.info(f"Loaded engine state from {filepath}")

    def reset_state(self) -> None:
        """Reset engine state."""
        self.market_state = MarketState()
        self.processing_stats = {
            "bars_processed": 0,
            "setups_generated": 0,
            "processing_time_ms": 0,
            "last_processed": None
        }

        logger.info("Reset engine state")

    def get_engine_info(self) -> dict[str, Any]:
        """Get engine information."""
        return {
            "config": asdict(self.config),
            "processing_stats": self.get_processing_stats(),
            "market_state": self.get_market_state_snapshot(),
            "detectors": detector_registry.get_detector_stats(),
            "hooks": {name: len(hooks) for name, hooks in self.hooks.items()}
        }

    def process_historical_bars(
        self,
        bars_5m: list[Bar],
        bars_1m: list[Bar]
    ) -> list[SetupProposal]:
        """
        Process historical bars and return all setup proposals.

        Args:
            bars_5m: List of 5-minute bars
            bars_1m: List of 1-minute bars

        Returns:
            List of all setup proposals
        """
        all_setups = []

        logger.info(f"Processing {len(bars_5m)} historical 5m bars")

        for i, bar_5m in enumerate(bars_5m):
            # Get 1m context window (5 bars before and after current 5m bar)
            context_start = max(0, i * 5 - 5)
            context_end = min(len(bars_1m), (i + 1) * 5 + 5)
            bars_1m_context = bars_1m[context_start:context_end]

            # Process bar
            setups = self.process_5m_bar_with_1m_context(bar_5m, bars_1m_context)
            all_setups.extend(setups)

            if i % 100 == 0:
                logger.info(f"Processed {i}/{len(bars_5m)} bars")

        logger.info(f"Completed processing, generated {len(all_setups)} setups")
        return all_setups

    def validate_setup(self, setup: SetupProposal) -> bool:
        """
        Validate a setup proposal.

        Args:
            setup: Setup proposal to validate

        Returns:
            True if setup is valid, False otherwise
        """
        # Basic validation
        if not setup.symbol:
            return False

        if setup.entry_price <= 0:
            return False

        if setup.stop_loss <= 0:
            return False

        if setup.take_profit <= 0:
            return False

        # Price relationship validation
        if setup.side == Side.BUY:
            if setup.entry_price <= setup.stop_loss:
                return False
            if setup.take_profit <= setup.entry_price:
                return False
        else:  # SELL
            if setup.entry_price >= setup.stop_loss:
                return False
            if setup.take_profit >= setup.entry_price:
                return False

        # Risk-reward validation
        if setup.risk_reward_ratio < 1.0:
            return False

        return True

    def filter_setups_by_quality(self, setups: list[SetupProposal], min_confidence: float = 0.7) -> list[SetupProposal]:
        """
        Filter setups by quality.

        Args:
            setups: List of setup proposals
            min_confidence: Minimum confidence threshold

        Returns:
            Filtered list of setup proposals
        """
        return [setup for setup in setups if setup.confidence >= min_confidence]

    def filter_setups_by_risk(self, setups: list[SetupProposal], max_risk_score: float = 5.0) -> list[SetupProposal]:
        """
        Filter setups by risk.

        Args:
            setups: List of setup proposals
            max_risk_score: Maximum risk score threshold

        Returns:
            Filtered list of setup proposals
        """
        filtered_setups = []

        for setup in setups:
            risk_score = getattr(setup, 'risk_assessment', {}).get('risk_score', 10.0)
            if risk_score <= max_risk_score:
                filtered_setups.append(setup)

        return filtered_setups
