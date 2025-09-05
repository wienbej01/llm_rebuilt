"""
Smoke test for the StrategyEngine.
"""
import csv
from datetime import datetime
from decimal import Decimal

from engine.strategy_engine import StrategyEngine
from engine.trading_types import Bar


def run_smoke_test():
    """Run the smoke test."""
    print("Starting smoke test...")

    # Load sample data
    bars = []
    with open("data/sample_es_5m.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bar = Bar(
                symbol="ES",
                timeframe="5m",
                session="ETH",
                venue="csv",
                timestamp=datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00")),
                open=Decimal(row["open"]),
                high=Decimal(row["high"]),
                low=Decimal(row["low"]),
                close=Decimal(row["close"]),
                volume=int(row["volume"]),
            )
            bars.append(bar)

    print(f"Loaded {len(bars)} bars.")

    # Initialize engine
    engine = StrategyEngine()

    # Process bars
    # The engine expects 5m bars and 1m context bars.
    # For this smoke test, I will pass the same 5m bars as 1m context.
    # This is not realistic, but it will test the engine's execution flow.
    all_setups = engine.process_historical_bars(bars, bars)

    print(f"Generated {len(all_setups)} setups.")

    for setup in all_setups:
        print(setup.model_dump_json(indent=2))

    print("Smoke test finished.")


if __name__ == "__main__":
    run_smoke_test()
