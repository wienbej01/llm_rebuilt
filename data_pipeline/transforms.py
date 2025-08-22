"""
Data transformation and feature engineering utilities.
Handles resampling, indicator calculations, and feature generation.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple
import logging
import math

import numpy as np
import pandas as pd
from numba import jit, float64, int64

from engine.types import Bar

logger = logging.getLogger(__name__)


class DataTransforms:
    """Data transformation and feature engineering utilities."""

    @staticmethod
    def resample_1m_to_5m(bars_1m: List[Bar]) -> List[Bar]:
        """
        Resample 1-minute bars to 5-minute bars.

        Args:
            bars_1m: List of 1-minute bars

        Returns:
            List of 5-minute bars
        """
        if not bars_1m:
            return []

        # Convert to DataFrame for easier manipulation
        df = DataTransforms._bars_to_dataframe(bars_1m)

        # Resample to 5-minute bars
        df_5m = df.resample('5min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        # Convert back to Bar objects
        bars_5m = []
        for timestamp, row in df_5m.iterrows():
            bar = Bar(
                timestamp=timestamp.to_pydatetime().replace(tzinfo=timezone.utc),
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=int(row['volume'])
            )
            bars_5m.append(bar)

        logger.info(f"Resampled {len(bars_1m)} 1m bars to {len(bars_5m)} 5m bars")
        return bars_5m

    @staticmethod
    def create_1m_context_windows(
        bars_5m: List[Bar],
        bars_1m: List[Bar],
        context_window: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Create sliding 1-minute context windows for each 5-minute bar.

        Args:
            bars_5m: List of 5-minute bars
            bars_1m: List of 1-minute bars
            context_window: Number of 1-minute bars in context window

        Returns:
            List of dictionaries containing 5m bar and its 1m context
        """
        if not bars_5m or not bars_1m:
            return []

        # Convert to DataFrames for easier time-based operations
        df_5m = DataTransforms._bars_to_dataframe(bars_5m)
        df_1m = DataTransforms._bars_to_dataframe(bars_1m)

        context_windows = []

        for idx, row_5m in df_5m.iterrows():
            bar_5m_timestamp = row_5m.name

            # Find 1-minute bars within context window before the 5-minute bar
            start_time = bar_5m_timestamp - timedelta(minutes=context_window)
            end_time = bar_5m_timestamp

            mask = (df_1m.index >= start_time) & (df_1m.index <= end_time)
            context_1m = df_1m.loc[mask]

            if len(context_1m) >= 5:  # At least 5 bars for meaningful context
                context_windows.append({
                    'bar_5m': DataTransforms._row_to_bar(row_5m),
                    'bars_1m_context': [DataTransforms._row_to_bar(row) for _, row in context_1m.iterrows()],
                    'context_size': len(context_1m)
                })

        logger.info(f"Created {len(context_windows)} context windows")
        return context_windows

    @staticmethod
    def calculate_indicators(bars: List[Bar]) -> Dict[str, List[float]]:
        """
        Calculate technical indicators for bars.

        Args:
            bars: List of bars

        Returns:
            Dictionary of indicator values
        """
        if not bars:
            return {}

        # Convert to DataFrame
        df = DataTransforms._bars_to_dataframe(bars)

        indicators = {}

        # Price-based indicators
        indicators['atr'] = DataTransforms._calculate_atr(df)
        indicators['vwap'] = DataTransforms._calculate_vwap(df)
        indicators['price_zscore'] = DataTransforms._calculate_price_zscore(df)

        # Volume indicators
        indicators['volume_sma'] = DataTransforms._calculate_volume_sma(df)
        indicators['volume_zscore'] = DataTransforms._calculate_volume_zscore(df)

        # Market structure indicators
        indicators['pdh'] = DataTransforms._calculate_pdh(df)  # Previous Day High
        indicators['pdl'] = DataTransforms._calculate_pdl(df)  # Previous Day Low
        indicators['onh'] = DataTransforms._calculate_onh(df)  # Overnight High
        indicators['onl'] = DataTransforms._calculate_onl(df)  # Overnight Low

        # Volatility indicators
        indicators['rolling_volatility'] = DataTransforms._calculate_rolling_volatility(df)

        logger.info(f"Calculated indicators for {len(bars)} bars")
        return indicators

    @staticmethod
    def join_1m_context_to_5m(
        bars_5m: List[Bar],
        bars_1m: List[Bar],
        context_window: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Join 1-minute context windows to 5-minute bars.

        Args:
            bars_5m: List of 5-minute bars
            bars_1m: List of 1-minute bars
            context_window: Size of 1-minute context window

        Returns:
            List of dictionaries with 5m bar and 1m context
        """
        return DataTransforms.create_1m_context_windows(bars_5m, bars_1m, context_window)

    @staticmethod
    def _bars_to_dataframe(bars: List[Bar]) -> pd.DataFrame:
        """Convert list of Bar objects to pandas DataFrame."""
        data = {
            'timestamp': [bar.timestamp for bar in bars],
            'open': [bar.open for bar in bars],
            'high': [bar.high for bar in bars],
            'low': [bar.low for bar in bars],
            'close': [bar.close for bar in bars],
            'volume': [bar.volume for bar in bars]
        }
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        return df

    @staticmethod
    def _row_to_bar(row: pd.Series) -> Bar:
        """Convert DataFrame row to Bar object."""
        return Bar(
            timestamp=row.name.to_pydatetime().replace(tzinfo=timezone.utc),
            open=float(row['open']),
            high=float(row['high']),
            low=float(row['low']),
            close=float(row['close']),
            volume=int(row['volume'])
        )

    @staticmethod
    def _calculate_atr(df: pd.DataFrame, period: int = 14) -> List[float]:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)

        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr.fillna(0).tolist()

    @staticmethod
    def _calculate_vwap(df: pd.DataFrame) -> List[float]:
        """Calculate Volume Weighted Average Price."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        return vwap.fillna(0).tolist()

    @staticmethod
    def _calculate_price_zscore(df: pd.DataFrame, window: int = 20) -> List[float]:
        """Calculate price z-score over rolling window."""
        zscore = (df['close'] - df['close'].rolling(window=window).mean()) / df['close'].rolling(window=window).std()
        return zscore.fillna(0).tolist()

    @staticmethod
    def _calculate_volume_sma(df: pd.DataFrame, period: int = 20) -> List[float]:
        """Calculate volume simple moving average."""
        sma = df['volume'].rolling(window=period).mean()
        return sma.fillna(0).tolist()

    @staticmethod
    def _calculate_volume_zscore(df: pd.DataFrame, window: int = 20) -> List[float]:
        """Calculate volume z-score over rolling window."""
        zscore = (df['volume'] - df['volume'].rolling(window=window).mean()) / df['volume'].rolling(window=window).std()
        return zscore.fillna(0).tolist()

    @staticmethod
    def _calculate_pdh(df: pd.DataFrame) -> List[float]:
        """Calculate Previous Day High."""
        # Group by date and calculate previous day high
        daily_high = df.groupby(df.index.date)['high'].max()
        pdh = daily_high.shift(1).reindex(df.index.date).fillna(method='ffill')
        return pdh.reindex(df.index).fillna(0).tolist()

    @staticmethod
    def _calculate_pdl(df: pd.DataFrame) -> List[float]:
        """Calculate Previous Day Low."""
        daily_low = df.groupby(df.index.date)['low'].min()
        pdl = daily_low.shift(1).reindex(df.index.date).fillna(method='ffill')
        return pdl.reindex(df.index).fillna(0).tolist()

    @staticmethod
    def _calculate_onh(df: pd.DataFrame) -> List[float]:
        """Calculate Overnight High (ETH session)."""
        # This is a simplified implementation
        # In practice, you'd need to know ETH session times
        onh = df['high'].rolling(window=24, min_periods=1).max()  # 24 hours
        return onh.tolist()

    @staticmethod
    def _calculate_onl(df: pd.DataFrame) -> List[float]:
        """Calculate Overnight Low (ETH session)."""
        onl = df['low'].rolling(window=24, min_periods=1).min()  # 24 hours
        return onl.tolist()

    @staticmethod
    def _calculate_rolling_volatility(df: pd.DataFrame, window: int = 20) -> List[float]:
        """Calculate rolling volatility (standard deviation of returns)."""
        returns = df['close'].pct_change()
        volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
        return volatility.fillna(0).tolist()


class FeatureEngineer:
    """Advanced feature engineering for trading signals."""

    @staticmethod
    def create_temporal_features(bars: List[Bar]) -> Dict[str, List[float]]:
        """Create time-based features."""
        if not bars:
            return {}

        timestamps = [bar.timestamp for bar in bars]

        features = {
            'hour_of_day': [ts.hour for ts in timestamps],
            'day_of_week': [ts.weekday() for ts in timestamps],
            'is_rth': [9 <= ts.hour <= 16 for ts in timestamps],  # Simplified RTH
            'minutes_since_midnight': [ts.hour * 60 + ts.minute for ts in timestamps]
        }

        return features

    @staticmethod
    def create_microstructure_features(bars: List[Bar]) -> Dict[str, List[float]]:
        """Create market microstructure features."""
        if not bars:
            return {}

        df = DataTransforms._bars_to_dataframe(bars)

        features = {}

        # Price movement patterns
        features['price_range_ratio'] = ((df['high'] - df['low']) / df['close'].shift(1)).fillna(0).tolist()
        features['gap_size'] = ((df['open'] - df['close'].shift(1)) / df['close'].shift(1)).fillna(0).tolist()

        # Volume patterns
        features['volume_price_trend'] = np.corrcoef(df['volume'], df['close'])[0, 1] if len(df) > 1 else 0.0
        features['relative_volume'] = (df['volume'] / df['volume'].rolling(20).mean()).fillna(1).tolist()

        # Volatility patterns
        features['atr_ratio'] = (DataTransforms._calculate_atr(df, 14) / df['close']).fillna(0).tolist()

        return features

    @staticmethod
    def create_smc_features(bars: List[Bar]) -> Dict[str, List[float]]:
        """Create Smart Money Concepts specific features."""
        if not bars:
            return {}

        df = DataTransforms._bars_to_dataframe(bars)

        features = {}

        # Fair Value Gap detection (simplified)
        fvg_threshold = 0.001  # 0.1% gap threshold
        price_gaps = abs(df['close'] - df['open'].shift(-1)) / df['close']
        features['has_fvg'] = (price_gaps > fvg_threshold).astype(int).tolist()

        # Swing detection (simplified)
        swing_threshold = 0.005  # 0.5% swing threshold
        is_swing_high = (df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1))
        is_swing_low = (df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(-1))
        features['is_swing_high'] = is_swing_high.astype(int).tolist()
        features['is_swing_low'] = is_swing_low.astype(int).tolist()

        # Market Structure Shift detection (simplified)
        price_change = abs(df['close'] - df['close'].shift(5)) / df['close'].shift(5)
        features['potential_mss'] = (price_change > 0.01).astype(int).tolist()  # 1% change

        return features


# Numba-optimized functions for performance-critical calculations
@jit(float64[:](float64[:], int64), nopython=True)
def calculate_sma_numba(values: np.ndarray, period: int) -> np.ndarray:
    """Numba-optimized Simple Moving Average."""
    n = len(values)
    result = np.zeros(n)

    for i in range(period - 1, n):
        result[i] = np.mean(values[i - period + 1:i + 1])

    return result


@jit(float64[:](float64[:], float64[:], float64[:], int64), nopython=True)
def calculate_atr_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Numba-optimized Average True Range."""
    n = len(high)
    tr = np.zeros(n)
    atr = np.zeros(n)

    # Calculate True Range
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i - 1])
        tr3 = abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)

    # Calculate ATR
    for i in range(period - 1, n):
        atr[i] = np.mean(tr[i - period + 1:i + 1])

    return atr