"""
Abstract base interface for market data providers.
All data providers must implement this interface for consistent data ingestion.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from datetime import UTC, date, datetime, timedelta
from enum import Enum
from typing import Any

from engine.types import Bar


class DataProviderType(str, Enum):
    """Types of data providers."""
    HISTORICAL = "historical"
    REALTIME = "realtime"
    SIMULATED = "simulated"


class ProviderError(Exception):
    """Base exception for data provider errors."""
    pass


class ConnectionError(ProviderError):
    """Raised when connection to data provider fails."""
    pass


class AuthenticationError(ProviderError):
    """Raised when authentication with data provider fails."""
    pass


class DataNotAvailableError(ProviderError):
    """Raised when requested data is not available."""
    pass


class RateLimitError(ProviderError):
    """Raised when rate limit is exceeded."""
    pass


class BaseDataProvider(ABC):
    """Abstract base class for all data providers."""

    def __init__(self, provider_type: DataProviderType, name: str):
        self.provider_type = provider_type
        self.name = name
        self.connected = False
        self._last_request_time: datetime | None = None

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the data provider."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the data provider."""
        pass

    @abstractmethod
    async def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start_date: date,
        end_date: date,
        **kwargs: Any
    ) -> list[Bar]:
        """
        Retrieve historical bar data.

        Args:
            symbol: Trading symbol (e.g., 'ES', 'NQ')
            timeframe: Bar timeframe (e.g., '1m', '5m', '1h')
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            **kwargs: Additional provider-specific parameters

        Returns:
            List of Bar objects sorted by timestamp

        Raises:
            ConnectionError: If not connected to provider
            AuthenticationError: If authentication fails
            DataNotAvailableError: If data is not available
            RateLimitError: If rate limit is exceeded
        """
        pass

    @abstractmethod
    async def get_available_symbols(self) -> list[str]:
        """Get list of available symbols from this provider."""
        pass

    @abstractmethod
    async def get_symbol_info(self, symbol: str) -> dict[str, Any]:
        """Get detailed information about a symbol."""
        pass

    @abstractmethod
    async def validate_symbol(self, symbol: str) -> bool:
        """Validate if a symbol is available and tradable."""
        pass

    async def stream_bars(
        self,
        symbol: str,
        timeframe: str,
        **kwargs: Any
    ) -> AsyncIterator[Bar]:
        """
        Stream real-time bar data (for real-time providers).

        Args:
            symbol: Trading symbol
            timeframe: Bar timeframe
            **kwargs: Additional provider-specific parameters

        Yields:
            Bar objects as they become available
        """
        raise NotImplementedError(f"{self.name} does not support streaming")

    def get_provider_info(self) -> dict[str, Any]:
        """Get information about this provider."""
        return {
            "name": self.name,
            "type": self.provider_type.value,
            "connected": self.connected,
            "last_request": self._last_request_time.isoformat() if self._last_request_time else None,
            "capabilities": self.get_capabilities()
        }

    def get_capabilities(self) -> dict[str, Any]:
        """Get provider capabilities."""
        return {
            "historical_data": True,
            "realtime_data": self.provider_type == DataProviderType.REALTIME,
            "streaming": hasattr(self, 'stream_bars') and callable(self.stream_bars),
            "supported_timeframes": self.get_supported_timeframes(),
            "max_lookback_days": self.get_max_lookback_days()
        }

    def get_supported_timeframes(self) -> list[str]:
        """Get list of supported timeframes."""
        return ["1m", "5m", "15m", "1h", "1d"]

    def get_max_lookback_days(self) -> int:
        """Get maximum lookback period in days."""
        return 365 * 5  # 5 years by default

    def update_last_request_time(self) -> None:
        """Update the last request timestamp."""
        from datetime import datetime
        self._last_request_time = datetime.now(UTC)

    def validate_timeframe(self, timeframe: str) -> bool:
        """Validate if timeframe is supported."""
        return timeframe in self.get_supported_timeframes()

    def validate_date_range(self, start_date: date, end_date: date) -> bool:
        """Validate date range."""
        if start_date >= end_date:
            return False

        from datetime import date
        max_lookback = self.get_max_lookback_days()
        if max_lookback:
            earliest_allowed = date.today() - timedelta(days=max_lookback)
            if start_date < earliest_allowed:
                return False

        return True

    async def __aenter__(self) -> BaseDataProvider:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()


class HistoricalDataProvider(BaseDataProvider):
    """Base class for historical data providers."""

    def __init__(self, name: str):
        super().__init__(DataProviderType.HISTORICAL, name)

    async def stream_bars(self, symbol: str, timeframe: str, **kwargs: Any) -> AsyncIterator[Bar]:
        """Historical providers don't support streaming."""
        raise NotImplementedError(f"{self.name} is a historical data provider and does not support streaming")


class RealtimeDataProvider(BaseDataProvider):
    """Base class for real-time data providers."""

    def __init__(self, name: str):
        super().__init__(DataProviderType.REALTIME, name)

    async def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start_date: date,
        end_date: date,
        **kwargs: Any
    ) -> list[Bar]:
        """Real-time providers can also provide historical data."""
        raise NotImplementedError(f"{self.name} does not implement historical data retrieval")


class SimulatedDataProvider(BaseDataProvider):
    """Base class for simulated data providers (for testing)."""

    def __init__(self, name: str):
        super().__init__(DataProviderType.SIMULATED, name)
