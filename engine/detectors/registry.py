"""
Detector registry for PSE-LLM trading system.
Manages and coordinates all trade setup detectors.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Callable, Type
import logging

from engine.types import SetupProposal, Bar
from engine.state import MarketState

logger = logging.getLogger(__name__)


class DetectorRegistry:
    """Registry for managing trade setup detectors."""

    def __init__(self):
        """Initialize detector registry."""
        self._detectors: Dict[str, Callable] = {}
        self._detector_configs: Dict[str, Dict[str, Any]] = {}
        self._enabled_detectors: Dict[str, bool] = {}

    def register_detector(
        self,
        name: str,
        detector_func: Callable,
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = True
    ) -> None:
        """
        Register a detector function.

        Args:
            name: Name of the detector
            detector_func: Detector function
            config: Configuration for the detector
            enabled: Whether the detector is enabled
        """
        self._detectors[name] = detector_func
        self._detector_configs[name] = config or {}
        self._enabled_detectors[name] = enabled

        logger.info(f"Registered detector: {name} (enabled: {enabled})")

    def unregister_detector(self, name: str) -> None:
        """
        Unregister a detector.

        Args:
            name: Name of the detector to unregister
        """
        if name in self._detectors:
            del self._detectors[name]
            del self._detector_configs[name]
            del self._enabled_detectors[name]
            logger.info(f"Unregistered detector: {name}")

    def get_detector(self, name: str) -> Optional[Callable]:
        """
        Get a detector function by name.

        Args:
            name: Name of the detector

        Returns:
            Detector function or None if not found
        """
        return self._detectors.get(name)

    def get_detector_config(self, name: str) -> Dict[str, Any]:
        """
        Get configuration for a detector.

        Args:
            name: Name of the detector

        Returns:
            Detector configuration
        """
        return self._detector_configs.get(name, {})

    def is_detector_enabled(self, name: str) -> bool:
        """
        Check if a detector is enabled.

        Args:
            name: Name of the detector

        Returns:
            True if enabled, False otherwise
        """
        return self._enabled_detectors.get(name, False)

    def enable_detector(self, name: str) -> None:
        """
        Enable a detector.

        Args:
            name: Name of the detector to enable
        """
        if name in self._enabled_detectors:
            self._enabled_detectors[name] = True
            logger.info(f"Enabled detector: {name}")

    def disable_detector(self, name: str) -> None:
        """
        Disable a detector.

        Args:
            name: Name of the detector to disable
        """
        if name in self._enabled_detectors:
            self._enabled_detectors[name] = False
            logger.info(f"Disabled detector: {name}")

    def list_detectors(self) -> List[str]:
        """
        Get list of all registered detector names.

        Returns:
            List of detector names
        """
        return list(self._detectors.keys())

    def list_enabled_detectors(self) -> List[str]:
        """
        Get list of enabled detector names.

        Returns:
            List of enabled detector names
        """
        return [name for name, enabled in self._enabled_detectors.items() if enabled]

    def update_detector_config(self, name: str, config: Dict[str, Any]) -> None:
        """
        Update configuration for a detector.

        Args:
            name: Name of the detector
            config: New configuration
        """
        if name in self._detector_configs:
            self._detector_configs[name].update(config)
            logger.info(f"Updated config for detector: {name}")

    def run_all_detectors(
        self,
        market_state: MarketState,
        bars_1m_window: List[Bar]
    ) -> List[SetupProposal]:
        """
        Run all enabled detectors and return setup proposals.

        Args:
            market_state: Current market state
            bars_1m_window: 1-minute bars for context

        Returns:
            List of setup proposals from all detectors
        """
        all_setups = []

        for detector_name in self.list_enabled_detectors():
            detector_func = self._detectors[detector_name]
            config = self._detector_configs[detector_name]

            try:
                # Run detector
                setups = detector_func(market_state, bars_1m_window, config)
                all_setups.extend(setups)

                logger.debug(f"Detector {detector_name} generated {len(setups)} setups")

            except Exception as e:
                logger.error(f"Error running detector {detector_name}: {e}")

        return all_setups

    def run_specific_detector(
        self,
        detector_name: str,
        market_state: MarketState,
        bars_1m_window: List[Bar]
    ) -> List[SetupProposal]:
        """
        Run a specific detector.

        Args:
            detector_name: Name of the detector to run
            market_state: Current market state
            bars_1m_window: 1-minute bars for context

        Returns:
            List of setup proposals from the detector
        """
        if detector_name not in self._detectors:
            logger.warning(f"Detector not found: {detector_name}")
            return []

        if not self._enabled_detectors[detector_name]:
            logger.warning(f"Detector disabled: {detector_name}")
            return []

        detector_func = self._detectors[detector_name]
        config = self._detector_configs[detector_name]

        try:
            setups = detector_func(market_state, bars_1m_window, config)
            logger.debug(f"Detector {detector_name} generated {len(setups)} setups")
            return setups

        except Exception as e:
            logger.error(f"Error running detector {detector_name}: {e}")
            return []

    def get_detector_stats(self) -> Dict[str, Any]:
        """
        Get statistics about detectors.

        Returns:
            Dictionary with detector statistics
        """
        return {
            "total_detectors": len(self._detectors),
            "enabled_detectors": len(self.list_enabled_detectors()),
            "disabled_detectors": len(self.list_detectors()) - len(self.list_enabled_detectors()),
            "detector_names": self.list_detectors(),
            "enabled_detector_names": self.list_enabled_detectors()
        }

    def load_config_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """
        Load detector configuration from dictionary.

        Args:
            config_dict: Configuration dictionary
        """
        for detector_name, detector_config in config_dict.items():
            if detector_name in self._detector_configs:
                self._detector_configs[detector_name].update(detector_config)

                # Enable/disable based on config
                if "enabled" in detector_config:
                    if detector_config["enabled"]:
                        self.enable_detector(detector_name)
                    else:
                        self.disable_detector(detector_name)

        logger.info("Loaded detector configuration from dictionary")

    def export_config_to_dict(self) -> Dict[str, Any]:
        """
        Export detector configuration to dictionary.

        Returns:
            Configuration dictionary
        """
        config_dict = {}

        for detector_name in self._detectors:
            config_dict[detector_name] = {
                **self._detector_configs[detector_name],
                "enabled": self._enabled_detectors[detector_name]
            }

        return config_dict


# Global detector registry instance
detector_registry = DetectorRegistry()


def get_detector_registry() -> DetectorRegistry:
    """
    Get the global detector registry instance.

    Returns:
        Global detector registry
    """
    return detector_registry


# Decorator for registering detectors
def register_detector(name: str, config: Optional[Dict[str, Any]] = None, enabled: bool = True):
    """
    Decorator for registering detector functions.

    Args:
        name: Name of the detector
        config: Configuration for the detector
        enabled: Whether the detector is enabled
    """
    def decorator(func):
        detector_registry.register_detector(name, func, config, enabled)
        return func
    return decorator