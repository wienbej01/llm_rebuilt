"""
Polygon.io provider implementation for market data ingestion.
Polygon provides comprehensive financial market data with REST and WebSocket APIs.
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


class PolygonProvider(HistoricalDataProvider):
    """Polygon.io data provider implementation."""

    def __init__(self, api_key: str):
        """
        Initialize Polygon provider.

        Args:
            api_key: Polygon API key
        """
        super().__init__("polygon")
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.session = None

    async def connect(self) -> None:
        """Establish connection to Polygon API."""
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
            async with self.session.get(f"{self.base_url}/v1/marketstatus/now") as response:
                if response.status == 401:
                    raise AuthenticationError("Invalid Polygon API key")
                elif response.status == 403:
                    raise AuthenticationError("Insufficient Polygon API permissions")
                elif response.status != 200:
                    raise ConnectionError(f"Failed to connect to Polygon: {response.status}")

            self.connected = True
            logger.info("Connected to Polygon API")

        except ImportError:
            raise ConnectionError("aiohttp is required for Polygon provider")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Polygon: {str(e)}")

    async def disconnect(self) -> None:
        """Close connection to Polygon API."""
        if self.session and not self.session.closed:
            await self.session.close()
        self.connected = False
        logger.info("Disconnected from Polygon API")

    async def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start_date: date,
        end_date: date,
        **kwargs: Any
    ) -> list[Bar]:
        """
        Retrieve historical bar data from Polygon.

        Args:
            symbol: Trading symbol (e.g., 'ES' for E-mini S&P 500)
            timeframe: Bar timeframe (e.g., '1m', '5m', '1h')
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            **kwargs: Additional parameters

        Returns:
            List of Bar objects
        """
        if not self.connected:
            raise ConnectionError("Not connected to Polygon")

        self.update_last_request_time()

        try:
            # Convert symbol to Polygon format if needed
            polygon_symbol = self._convert_symbol(symbol)

            # Convert timeframe to Polygon format
            multiplier, timespan = self._convert_timeframe(timeframe)

            # Build request parameters
            params = {
                "adjusted": "true",
                "sort": "asc",
                "limit": 50000  # Maximum allowed by Polygon
            }

            # Make API request
            url = f"{self.base_url}/v2/aggs/ticker/{polygon_symbol}/range/{multiplier}/{timespan}/{start_date.isoformat()}/{end_date.isoformat()}"

            async with self.session.get(url, params=params) as response:
                if response.status == 401:
                    raise AuthenticationError("Invalid Polygon API key")
                elif response.status == 403:
                    raise AuthenticationError("Insufficient permissions for this data")
                elif response.status == 404:
                    raise DataNotAvailableError(f"Data not available for {symbol}")
                elif response.status == 429:
                    raise RateLimitError("Polygon rate limit exceeded")
                elif response.status != 200:
                    raise DataNotAvailableError(f"Polygon API error: {response.status}")

                data = await response.json()

            # Parse response and create Bar objects
            bars = []
            if "results" in data:
                for record in data["results"]:
                    bar = self._parse_polygon_record(record, symbol)
                    if bar:
                        bars.append(bar)

            logger.info(f"Retrieved {len(bars)} bars for {symbol} from Polygon")
            return bars

        except Exception as e:
            logger.error(f"Error retrieving data from Polygon: {str(e)}")
            raise DataNotAvailableError(f"Failed to retrieve data: {str(e)}")

    def _convert_symbol(self, symbol: str) -> str:
        """Convert internal symbol format to Polygon format."""
        # Polygon uses different symbol formats for different asset classes
        symbol_mappings = {
            "ES": "ES",  # E-mini S&P 500 futures
            "NQ": "NQ",  # E-mini NASDAQ 100 futures
            "RTY": "RTY",  # E-mini Russell 2000 futures
            "GC": "GC",  # Gold futures
            "SI": "SI",  # Silver futures
            "CL": "CL",  # Crude Oil futures
        }
        return symbol_mappings.get(symbol, symbol)

    def _convert_timeframe(self, timeframe: str) -> tuple[int, str]:
        """Convert internal timeframe format to Polygon format."""
        conversion_map = {
            "1m": (1, "minute"),
            "5m": (5, "minute"),
            "15m": (15, "minute"),
            "1h": (1, "hour"),
            "1d": (1, "day")
        }
        return conversion_map.get(timeframe, (1, "minute"))

    def _parse_polygon_record(self, record: dict[str, Any], symbol: str) -> Bar | None:
        """Parse a Polygon record into a Bar object."""
        try:
            # Polygon uses Unix milliseconds
            timestamp_ms = record.get("t", 0)
            timestamp = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)

            # Extract OHLCV data
            open_price = float(record.get("o", 0))
            high_price = float(record.get("h", 0))
            low_price = float(record.get("l", 0))
            close_price = float(record.get("c", 0))
            volume = int(record.get("v", 0))

            return Bar(
                timestamp=timestamp,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume
            )

        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse Polygon record: {str(e)}")
            return None

    async def get_available_symbols(self) -> list[str]:
        """Get list of available symbols from Polygon."""
        if not self.connected:
            raise ConnectionError("Not connected to Polygon")

        try:
            # Get supported tickers (this is a simplified implementation)
            # In practice, you might want to cache this or use specific endpoints
            return ["ES", "NQ", "RTY", "GC", "SI", "CL"]  # Common futures

        except Exception as e:
            logger.error(f"Error retrieving symbols from Polygon: {str(e)}")
            return []

    async def get_symbol_info(self, symbol: str) -> dict[str, Any]:
        """Get detailed information about a symbol."""
        if not self.connected:
            raise ConnectionError("Not connected to Polygon")

        try:
            polygon_symbol = self._convert_symbol(symbol)
            async with self.session.get(f"{self.base_url}/v3/reference/tickers/{polygon_symbol}") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("results", {})
                else:
                    return {}

        except Exception as e:
            logger.error(f"Error retrieving symbol info from Polygon: {str(e)}")
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
        return 365 * 2  # 2 years for free tier, more for paid tiers

    def get_capabilities(self) -> dict[str, Any]:
        """Get provider capabilities."""
        base_capabilities = super().get_capabilities()
        base_capabilities.update({
            "polygon_specific": {
                "supports_options": True,
                "supports_crypto": True,
                "supports_forex": True,
                "has_news_api": True,
                "has_reference_data": True
            }
        })
        return base_capabilities
