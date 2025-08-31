"""
DataBento provider implementation for market data ingestion.
DataBento provides high-quality financial market data with low latency.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from typing import Any

from engine.trading_types import Bar

from .base import (
    AuthenticationError,
    ConnectionError,
    DataNotAvailableError,
    HistoricalDataProvider,
    RateLimitError,
)

logger = logging.getLogger(__name__)


class DatabentoProvider(HistoricalDataProvider):
    """DataBento data provider implementation."""

    def __init__(self, api_key: str, dataset: str = "GLBX.MDP3"):
        """
        Initialize DataBento provider.

        Args:
            api_key: DataBento API key
            dataset: Dataset to use (default: GLBX.MDP3 for global equities)
        """
        super().__init__("databento")
        self.api_key = api_key
        self.dataset = dataset
        self.base_url = "https://hist.databento.com/v0"
        self.session = None

    async def connect(self) -> None:
        """Establish connection to DataBento API."""
        try:
            import aiohttp

            if self.session is None or self.session.closed:
                self.session = aiohttp.ClientSession(
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                )

            # Test connection with a simple API call
            async with self.session.get(f"{self.base_url}/metadata.datasets") as response:
                if response.status == 401:
                    raise AuthenticationError("Invalid DataBento API key")
                elif response.status != 200:
                    raise ConnectionError(f"Failed to connect to DataBento: {response.status}")

            self.connected = True
            logger.info(f"Connected to DataBento API with dataset {self.dataset}")

        except ImportError:
            raise ConnectionError("aiohttp is required for DataBento provider")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to DataBento: {str(e)}")

    async def disconnect(self) -> None:
        """Close connection to DataBento API."""
        if self.session and not self.session.closed:
            await self.session.close()
        self.connected = False
        logger.info("Disconnected from DataBento API")

    async def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start_date: date,
        end_date: date,
        **kwargs: Any
    ) -> list[Bar]:
        """
        Retrieve historical bar data from DataBento.

        Args:
            symbol: Trading symbol (e.g., 'ESZ4' for ES Dec 2024)
            timeframe: Bar timeframe (e.g., '1m', '5m')
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            **kwargs: Additional parameters

        Returns:
            List of Bar objects
        """
        if not self.connected:
            raise ConnectionError("Not connected to DataBento")

        self.update_last_request_time()

        try:
            # Convert timeframe to DataBento format
            db_timeframe = self._convert_timeframe(timeframe)

            # Build request parameters
            params = {
                "dataset": self.dataset,
                "symbols": symbol,
                "schema": "ohlcv-1m" if timeframe == "1m" else "ohlcv-1m",  # DataBento uses 1m base
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "encoding": "json"
            }

            # Add aggregation for non-1m timeframes
            if timeframe != "1m":
                params["agg"] = db_timeframe

            # Make API request
            async with self.session.get(f"{self.base_url}/timeseries.get_range", params=params) as response:
                if response.status == 401:
                    raise AuthenticationError("Invalid DataBento API key")
                elif response.status == 404:
                    raise DataNotAvailableError(f"Data not available for {symbol}")
                elif response.status == 429:
                    raise RateLimitError("DataBento rate limit exceeded")
                elif response.status != 200:
                    raise DataNotAvailableError(f"DataBento API error: {response.status}")

                data = await response.json()

            # Parse response and create Bar objects
            bars = []
            if "data" in data:
                for record in data["data"]:
                    bar = self._parse_databento_record(record, symbol, timeframe)
                    if bar:
                        bars.append(bar)

            logger.info(f"Retrieved {len(bars)} bars for {symbol} from DataBento")
            return bars

        except Exception as e:
            logger.error(f"Error retrieving data from DataBento: {str(e)}")
            raise DataNotAvailableError(f"Failed to retrieve data: {str(e)}")

    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert internal timeframe format to DataBento format."""
        conversion_map = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "1h": "1h",
            "1d": "1d"
        }
        return conversion_map.get(timeframe, "1m")

    def _parse_databento_record(self, record: dict[str, Any], symbol: str, timeframe: str) -> Bar | None:
        """Parse a DataBento record into a Bar object."""
        try:
            # DataBento uses Unix nanoseconds
            timestamp_ns = record.get("ts_event", 0)
            timestamp = datetime.fromtimestamp(timestamp_ns / 1e9, tz=UTC)

            # Extract OHLCV data
            open_price = float(record.get("open", 0))
            high_price = float(record.get("high", 0))
            low_price = float(record.get("low", 0))
            close_price = float(record.get("close", 0))
            volume = int(record.get("volume", 0))

            return Bar(
                symbol=symbol,
                timeframe=timeframe,
                session="ETH",  # Databento is typically ETH
                venue=self.name,
                timestamp=timestamp,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume
            )

        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse DataBento record: {str(e)}")
            return None

    async def get_available_symbols(self) -> list[str]:
        """Get list of available symbols from DataBento."""
        if not self.connected:
            raise ConnectionError("Not connected to DataBento")

        try:
            # Get dataset metadata
            async with self.session.get(f"{self.base_url}/metadata.datasets") as response:
                if response.status == 200:
                    data = await response.json()
                    # Extract symbols from dataset info
                    # This is a simplified implementation
                    return ["ES", "NQ", "RTY", "GC", "SI", "CL"]  # Common futures
                else:
                    raise DataNotAvailableError("Failed to retrieve symbol list")

        except Exception as e:
            logger.error(f"Error retrieving symbols from DataBento: {str(e)}")
            return []

    async def get_symbol_info(self, symbol: str) -> dict[str, Any]:
        """Get detailed information about a symbol."""
        if not self.connected:
            raise ConnectionError("Not connected to DataBento")

        try:
            # Get instrument metadata
            params = {"dataset": self.dataset, "symbols": symbol}
            async with self.session.get(f"{self.base_url}/metadata.list_instruments", params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", {}).get(symbol, {})
                else:
                    return {}

        except Exception as e:
            logger.error(f"Error retrieving symbol info from DataBento: {str(e)}")
            return {}

    async def validate_symbol(self, symbol: str) -> bool:
        """Validate if a symbol is available and tradable."""
        try:
            info = await self.get_symbol_info(symbol)
            return bool(info)
        except Exception:
            return False

    def get_supported_timeframes(self) -> list[str]:
        """Get list of supported timeframes."""
        return ["1m", "5m", "15m", "1h", "1d"]

    def get_max_lookback_days(self) -> int:
        """Get maximum lookback period in days."""
        return 365 * 10  # 10 years for DataBento
