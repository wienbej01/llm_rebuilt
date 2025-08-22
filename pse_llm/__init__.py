"""Python Strategy Engine with LLM Integration (PSE-LLM) - A quantitative trading system."""

__version__ = "1.0.0"
__author__ = "Trading Team"
__description__ = "Advanced trading strategy engine with LLM integration for optimal trade decision making"

from . import engine, llm, execution, backtest, cli, data_pipeline

__all__ = ["engine", "llm", "execution", "backtest", "cli", "data_pipeline"]