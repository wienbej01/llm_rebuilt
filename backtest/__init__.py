"""
Backtesting framework for PSE-LLM.

Provides tools to run historical simulations, calculate performance metrics,
and generate reports.
"""

from .engine import BacktestEngine, BacktestConfig, BacktestResult
from .metrics import calculate_metrics
from .reports import generate_html_report

__all__ = [
    "BacktestEngine",
    "BacktestConfig",
    "BacktestResult",
    "calculate_metrics",
    "generate_html_report",
]
