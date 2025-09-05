"""
IBKR historical data provider implementation.
Uses IBKR TWS API for historical market data with RTH/ETH session support.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, date
from typing import Any

from engine.trading_types import Bar

from .base import (
    ConnectionError,
    DataNotAvailableError,
    HistoricalDataProvider,
)

logger = logging.getLogger(__name__)


class IBKRHistoricalProvider(HistoricalDataProvider):
    """IBKR historical data provider implementation."""

    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1):
        """
        Initialize IBKR historical provider.

        Args:
            host: TWS/Gateway host
            port: TWS/Gateway port (7497 for TWS paper, 7496 for TWS live, 4002 for Gateway paper, 4001 for Gateway live)
            client_id: Client ID for connection
        """
        super().__init__("ibkr_hist")
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = None
        self.connected = False

    async def connect(self) -> None:
        """Establish connection to IBKR TWS/Gateway."""
        try:
            from ib_insync import IB, util

            if self.ib is None:
                self.ib = IB()

            # Connect to IBKR
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.ib.connect(self.host, self.port, clientId=self.client_id)
            )

            # Wait for connection
            await asyncio.sleep(2)

            if not self.ib.isConnected():
                raise ConnectionError("Failed to connect to IBKR TWS/Gateway")

            self.connected = True
            logger.info(f"Connected to IBKR at {self.host}:{self.port}")

        except ImportError:
            raise ConnectionError("ib_insync is required for IBKR provider")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to IBKR: {str(e)}")

    async def disconnect(self) -> None:
        """Close connection to IBKR TWS/Gateway."""
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
        self.connected = False
        logger.info("Disconnected from IBKR")

    async def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start_date: date,
        end_date: date,
        **kwargs: Any
    ) -> list[Bar]:
        """
        Retrieve historical bar data from IBKR.

        Args:
            symbol: Trading symbol (e.g., 'ES', 'NQ')
            timeframe: Bar timeframe (e.g., '1m', '5m')
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            **kwargs: Additional parameters (e.g., session='RTH')

        Returns:
            List of Bar objects
        """
        if not self.connected or not self.ib:
            raise ConnectionError("Not connected to IBKR")

        self.update_last_request_time()

        try:
            # Create IBKR contract
            contract = self._create_contract(symbol)

            # Convert timeframe to IBKR format
            duration, bar_size = self._convert_timeframe(timeframe, start_date, end_date)

            # Get session type (RTH or ETH)
            session = kwargs.get('session', 'RTH')

            # Request historical data
            bars = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.ib.reqHistoricalData(
                    contract,
                    endDateTime=end_date.isoformat() + " 23:59:59",
                    durationStr=duration,
                    barSizeSetting=bar_size,
                    whatToShow="TRADES",
                    useRTH=session == "RTH",
                    formatDate=1
                )
            )

            # Convert IBKR bars to our Bar objects
            result_bars = []
            for ib_bar in bars:
                bar = self._convert_ibkr_bar(ib_bar, symbol, timeframe, session)
                if bar:
                    result_bars.append(bar)

            logger.info(f"Retrieved {len(result_bars)} bars for {symbol} from IBKR")
            return result_bars

        except Exception as e:
            logger.error(f"Error retrieving data from IBKR: {str(e)}")
            raise DataNotAvailableError(f"Failed to retrieve data: {str(e)}")

    def _create_contract(self, symbol: str) -> Any:
        """Create IBKR contract for the given symbol."""
        from ib_insync import Contract

        # Map symbols to IBKR contracts
        # Use continuous futures for simplicity
        exchange_map = {
            "ES": "CME",
            "NQ": "CME",
            "RTY": "CME",
            "GC": "COMEX",
            "SI": "COMEX",
            "CL": "NYMEX",
        }
        exchange = exchange_map.get(symbol, "SMART")

        return Contract(symbol=symbol, secType="CONTFUT", exchange=exchange, currency="USD")

    def _convert_timeframe(self, timeframe: str, start_date: date, end_date: date) -> tuple[str, str]:
        """Convert internal timeframe format to IBKR format."""
        # Calculate duration
        days = (end_date - start_date).days
        if days <= 1:
            duration = "1 D"
        elif days <= 7:
            duration = "1 W"
        elif days <= 30:
            duration = "1 M"
        else:
            duration = f"{days} D"

        # Convert timeframe
        timeframe_map = {
            "1m": "1 min",
            "5m": "5 mins",
            "15m": "15 mins",
            "1h": "1 hour",
            "1d": "1 day"
        }

        bar_size = timeframe_map.get(timeframe, "5 mins")
        return duration, bar_size

    def _convert_ibkr_bar(self, ib_bar: Any, symbol: str, timeframe: str, session: str) -> Bar | None:
        """Convert IBKR bar to our Bar object."""
        try:
            # IBKR uses timezone-aware datetime
            timestamp = ib_bar.date.replace(tzinfo=UTC) if ib_bar.date.tzinfo is None else ib_bar.date.astimezone(UTC)

            return Bar(
                symbol=symbol,
                timeframe=timeframe,
                session=session,
                venue=self.name,
                timestamp=timestamp,
                open=float(ib_bar.open),
                high=float(ib_bar.high),
                low=float(ib_bar.low),
                close=float(ib_bar.close),
                volume=int(ib_bar.volume)
            )

        except (AttributeError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse IBKR bar: {str(e)}")
            return None

    async def get_available_symbols(self) -> list[str]:
        """Get list of available symbols from IBKR."""
        if not self.connected:
            raise ConnectionError("Not connected to IBKR")

        try:
            # Common futures symbols available on IBKR
            return ["ES", "NQ", "RTY", "GC", "SI", "CL", "ZN", "ZF"]

        except Exception as e:
            logger.error(f"Error retrieving symbols from IBKR: {str(e)}")
            return []

    async def get_symbol_info(self, symbol: str) -> dict[str, Any]:
        """Get detailed information about a symbol."""
        if not self.connected or not self.ib:
            raise ConnectionError("Not connected to IBKR")

        try:
            contract = self._create_contract(symbol)

            # Get contract details
            details = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.ib.reqContractDetails(contract)
            )

            if details:
                detail = details[0]
                return {
                    "symbol": detail.contract.symbol,
                    "exchange": detail.contract.exchange,
                    "currency": detail.contract.currency,
                    "secType": detail.contract.secType,
                    "longName": detail.longName,
                    "minTick": detail.minTick,
                    "priceMagnifier": detail.priceMagnifier,
                    "tradingHours": detail.tradingHours,
                    "liquidHours": detail.liquidHours
                }
            else:
                return {}

        except Exception as e:
            logger.error(f"Error retrieving symbol info from IBKR: {str(e)}")
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
        return 365 * 5  # 5 years for IBKR historical data

    def get_capabilities(self) -> dict[str, Any]:
        """Get provider capabilities."""
        base_capabilities = super().get_capabilities()
        base_capabilities.update({
            "ibkr_specific": {
                "supports_futures": True,
                "supports_options": True,
                "supports_stocks": True,
                "supports_forex": True,
                "has_realtime_data": True,
                "has_news_data": True,
                "has_fundamentals": True,
                "rth_eth_support": True
            }
        })
        return base_capabilities
