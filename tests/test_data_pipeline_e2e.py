"""
End-to-end tests for the data pipeline.
Verifies the flow from provider ingestion -> transformation -> feature store persistence.
"""

import asyncio
import shutil
from datetime import datetime, date, timezone, timedelta
from decimal import Decimal
from pathlib import Path
from typing import List

import pytest

from data_pipeline.feature_store import FeatureStore, FeatureStoreConfig
from data_pipeline.providers.polygon import PolygonProvider
from data_pipeline.transforms import DataTransforms
from engine.types import Bar, SessionType

# --- Test Fixtures ---

@pytest.fixture
def feature_store() -> FeatureStore:
    """Provides a temporary feature store for testing."""
    temp_path = Path("./test_feature_store_e2e")
    if temp_path.exists():
        shutil.rmtree(temp_path)
        
    config = FeatureStoreConfig(
        base_path=str(temp_path),
        enable_duckdb=False  # Disable DuckDB for simpler file-based testing
    )
    fs = FeatureStore(config)
    yield fs
    # Teardown
    if temp_path.exists():
        shutil.rmtree(temp_path)

# --- Helper Functions ---

def create_mock_1m_bars(symbol: str, start_time: datetime, num_bars: int) -> List[Bar]:
    """Creates a list of mock 1-minute Bar objects for testing."""
    bars = []
    for i in range(num_bars):
        ts = start_time + timedelta(minutes=i)
        price = Decimal('100') + Decimal(i) * Decimal('0.1')
        bars.append(
            Bar(
                symbol=symbol,
                timeframe='1m',
                timestamp=ts,
                session=SessionType.RTH,
                venue='TEST',
                open=price,
                high=price + Decimal('0.5'),
                low=price - Decimal('0.5'),
                close=price + Decimal('0.2'),
                volume=100 + i
            )
        )
    return bars

# --- E2E Test ---

@pytest.mark.asyncio
async def test_data_pipeline_e2e(mocker, feature_store: FeatureStore):
    """
    Verifies the end-to-end data pipeline flow:
    1. Ingestion (mocked) -> 2. Transformation -> 3. Storage -> 4. Read-back
    """
    # --- Test Setup ---
    symbol = "ES_TEST"
    start_date = date(2023, 1, 1)
    end_date = date(2023, 1, 1)
    start_time = datetime(2023, 1, 1, 9, 30, tzinfo=timezone.utc)
    
    mock_bars_1m = create_mock_1m_bars(symbol, start_time, 10)

    mocker.patch.object(
        PolygonProvider, 
        'get_bars', 
        return_value=asyncio.Future()
    ).return_value.set_result(mock_bars_1m)

    provider = PolygonProvider(api_key="fake_key")

    # 1. INGESTION (from mocked provider)
    ingested_bars_1m = await provider.get_bars(symbol, '1m', start_date, end_date)
    assert len(ingested_bars_1m) == 10

    # 2. TRANSFORMATION (Resampling)
    resampled_bars_5m = DataTransforms.resample_1m_to_5m(ingested_bars_1m)
    assert len(resampled_bars_5m) == 2
    bar_5m_1 = resampled_bars_5m[0]
    assert bar_5m_1.open == Decimal('100.0')
    assert bar_5m_1.high == Decimal('100.9')
    assert bar_5m_1.low == Decimal('99.5')
    assert bar_5m_1.close == Decimal('100.6')
    assert bar_5m_1.volume == 510

    # 3. STORAGE
    await feature_store.store_bars(symbol, '1m', ingested_bars_1m)
    await feature_store.store_bars(symbol, '5m', resampled_bars_5m)

    # 4. VERIFICATION (Read-back)
    loaded_bars_1m = await feature_store.load_bars(symbol, '1m', start_date, end_date)
    loaded_bars_5m = await feature_store.load_bars(symbol, '5m', start_date, end_date)

    assert loaded_bars_1m == ingested_bars_1m
    assert loaded_bars_5m == resampled_bars_5m
