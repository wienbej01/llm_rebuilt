"""
Performance metrics for backtesting.
"""
from __future__ import annotations

from typing import List, Dict, Any, Tuple
from datetime import datetime
from decimal import Decimal

import numpy as np

from execution.sim_broker import Position


def calculate_metrics(
    positions: List[Position],
    equity_curve: List[Tuple[datetime, Decimal]]
) -> Dict[str, Any]:
    """Calculate performance metrics from a backtest."""
    if not positions:
        return {}

    realized_pnl_list = [p.realized_pnl for p in positions if p.trades_count > 0 and p.quantity == 0]
    if not realized_pnl_list:
        return {"total_trades": 0}

    total_trades = len(realized_pnl_list)
    total_pnl = sum(realized_pnl_list)
    winning_trades = len([pnl for pnl in realized_pnl_list if pnl > 0])
    losing_trades = total_trades - winning_trades

    gross_profit = sum(pnl for pnl in realized_pnl_list if pnl > 0)
    gross_loss = abs(sum(pnl for pnl in realized_pnl_list if pnl < 0))

    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float('inf')

    # Max Drawdown
    equity_values = np.array([float(e[1]) for e in equity_curve])
    peak = np.maximum.accumulate(equity_values)
    drawdown = (peak - equity_values) / peak
    max_drawdown = float(np.max(drawdown)) if len(drawdown) > 0 and np.any(peak > 0) else 0.0

    # Sharpe Ratio (simplified, assumes daily returns)
    returns = np.diff(equity_values) / equity_values[:-1] if len(equity_values) > 1 else np.array([])
    sharpe_ratio = float(np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 and len(returns) > 0 else 0.0

    return {
        "total_trades": total_trades,
        "total_pnl": float(total_pnl),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
    }
