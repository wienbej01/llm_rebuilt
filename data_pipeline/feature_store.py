"""
Feature store implementation with Parquet and DuckDB support.
Handles data persistence, versioning, and efficient querying.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, date, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import json
import hashlib
import logging
import shutil

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

from engine.types import Bar

logger = logging.getLogger(__name__)


class FeatureStoreConfig(BaseModel):
    """Configuration for feature store."""
    base_path: str = "data/features"
    compression: str = "snappy"
    row_group_size: int = 100000
    enable_duckdb: bool = True
    duckdb_path: str = "data/features.duckdb"
    enable_versioning: bool = True
    max_versions: int = 10


class DataVersion(BaseModel):
    """Data version information."""
    version: str
    timestamp: datetime
    checksum: str
    record_count: int
    schema_hash: str
    metadata: Dict[str, Any]


class FeatureStore:
    """Feature store with Parquet and DuckDB backend."""

    def __init__(self, config: FeatureStoreConfig):
        self.config = config
        self.base_path = Path(config.base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.duckdb_conn = None
        if config.enable_duckdb:
            self._init_duckdb()

        logger.info(f"Initialized feature store at {self.base_path}")

    def _init_duckdb(self) -> None:
        """Initialize DuckDB connection."""
        try:
            import duckdb

            self.duckdb_conn = duckdb.connect(str(self.config.duckdb_path))
            self._create_duckdb_tables()
            logger.info("DuckDB connection initialized")

        except ImportError:
            logger.warning("DuckDB not available, falling back to Parquet-only mode")
            self.config.enable_duckdb = False

    def _create_duckdb_tables(self) -> None:
        """Create DuckDB tables for metadata and indexing."""
        if not self.duckdb_conn:
            return

        # Create versions table
        self.duckdb_conn.execute("""
            CREATE TABLE IF NOT EXISTS data_versions (
                symbol TEXT,
                timeframe TEXT,
                date DATE,
                version TEXT,
                timestamp TIMESTAMP,
                checksum TEXT,
                record_count INTEGER,
                schema_hash TEXT,
                metadata JSON,
                PRIMARY KEY (symbol, timeframe, date, version)
            )
        """)

        # Create file index table
        self.duckdb_conn.execute("""
            CREATE TABLE IF NOT EXISTS file_index (
                symbol TEXT,
                timeframe TEXT,
                date DATE,
                file_path TEXT,
                version TEXT,
                created_at TIMESTAMP,
                PRIMARY KEY (symbol, timeframe, date, version)
            )
        """)

    async def store_bars(
        self,
        symbol: str,
        timeframe: str,
        bars: List[Bar],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store bars data with versioning and metadata.

        Args:
            symbol: Trading symbol
            timeframe: Bar timeframe
            bars: List of bars to store
            metadata: Additional metadata

        Returns:
            Version string of stored data
        """
        if not bars:
            raise ValueError("No bars to store")

        # Generate version
        version = self._generate_version(bars)

        # Convert bars to DataFrame
        df = self._bars_to_dataframe(bars)

        # Add metadata columns
        df['symbol'] = symbol
        df['timeframe'] = timeframe
        df['version'] = version

        # Store as Parquet
        file_path = self._get_file_path(symbol, timeframe, bars[0].timestamp.date(), version)
        await self._store_parquet(df, file_path)

        # Store version metadata
        await self._store_version_metadata(symbol, timeframe, bars, version, metadata or {})

        # Update DuckDB index if available
        if self.duckdb_conn:
            await self._update_duckdb_index(symbol, timeframe, bars[0].timestamp.date(), file_path, version)

        logger.info(f"Stored {len(bars)} bars for {symbol}_{timeframe} version {version}")
        return version

    async def load_bars(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        version: Optional[str] = None
    ) -> List[Bar]:
        """
        Load bars data from feature store.

        Args:
            symbol: Trading symbol
            timeframe: Bar timeframe
            start_date: Start date filter
            end_date: End date filter
            version: Specific version to load

        Returns:
            List of bars
        """
        # Find relevant files
        files = await self._find_data_files(symbol, timeframe, start_date, end_date, version)

        if not files:
            return []

        # Load and combine data
        dfs = []
        for file_path in files:
            df = await self._load_parquet(file_path)
            if df is not None:
                dfs.append(df)

        if not dfs:
            return []

        # Combine DataFrames
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df.sort_values('timestamp', inplace=True)

        # Convert to bars
        bars = []
        for _, row in combined_df.iterrows():
            bar = Bar(
                timestamp=row['timestamp'].to_pydatetime().replace(tzinfo=timezone.utc),
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=int(row['volume'])
            )
            bars.append(bar)

        logger.info(f"Loaded {len(bars)} bars for {symbol}_{timeframe}")
        return bars

    async def store_features(
        self,
        symbol: str,
        feature_name: str,
        features: Dict[str, List[float]],
        timestamps: List[datetime],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store calculated features.

        Args:
            symbol: Trading symbol
            feature_name: Name of the feature set
            features: Dictionary of feature arrays
            timestamps: Corresponding timestamps
            metadata: Additional metadata

        Returns:
            Version string of stored features
        """
        if not features or not timestamps:
            raise ValueError("No features or timestamps to store")

        # Create DataFrame
        df_data = {'timestamp': timestamps}
        df_data.update(features)
        df = pd.DataFrame(df_data)

        # Add metadata
        df['symbol'] = symbol
        df['feature_name'] = feature_name

        # Generate version
        version = self._generate_version_from_data(df.to_dict('records'))

        df['version'] = version

        # Store as Parquet
        file_path = self._get_feature_file_path(symbol, feature_name, timestamps[0].date(), version)
        await self._store_parquet(df, file_path)

        # Store version metadata
        await self._store_feature_version_metadata(symbol, feature_name, version, metadata or {})

        logger.info(f"Stored features {feature_name} for {symbol} version {version}")
        return version

    async def load_features(
        self,
        symbol: str,
        feature_name: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        version: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Load features from feature store.

        Args:
            symbol: Trading symbol
            feature_name: Name of the feature set
            start_date: Start date filter
            end_date: End date filter
            version: Specific version to load

        Returns:
            Dictionary of feature arrays
        """
        # Find relevant files
        files = await self._find_feature_files(symbol, feature_name, start_date, end_date, version)

        if not files:
            return {}

        # Load and combine data
        dfs = []
        for file_path in files:
            df = await self._load_parquet(file_path)
            if df is not None:
                dfs.append(df)

        if not dfs:
            return {}

        # Combine DataFrames
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df.sort_values('timestamp', inplace=True)

        # Extract features
        features = {}
        exclude_cols = {'timestamp', 'symbol', 'feature_name', 'version'}
        for col in combined_df.columns:
            if col not in exclude_cols:
                features[col] = combined_df[col].tolist()

        logger.info(f"Loaded features {feature_name} for {symbol}")
        return features

    def _bars_to_dataframe(self, bars: List[Bar]) -> pd.DataFrame:
        """Convert bars to DataFrame."""
        data = {
            'timestamp': [bar.timestamp for bar in bars],
            'open': [bar.open for bar in bars],
            'high': [bar.high for bar in bars],
            'low': [bar.low for bar in bars],
            'close': [bar.close for bar in bars],
            'volume': [bar.volume for bar in bars]
        }
        return pd.DataFrame(data)

    def _generate_version(self, bars: List[Bar]) -> str:
        """Generate version string based on bar data."""
        # Create hash of the data
        data_str = json.dumps([bar.model_dump() for bar in bars], sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()[:12]

    def _generate_version_from_data(self, data: List[Dict]) -> str:
        """Generate version string from data."""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()[:12]

    def _get_file_path(self, symbol: str, timeframe: str, date_val: date, version: str) -> Path:
        """Get file path for storing data."""
        path = self.base_path / symbol / timeframe / date_val.isoformat() / f"{version}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _get_feature_file_path(self, symbol: str, feature_name: str, date_val: date, version: str) -> Path:
        """Get file path for storing features."""
        path = self.base_path / "features" / symbol / feature_name / date_val.isoformat() / f"{version}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    async def _store_parquet(self, df: pd.DataFrame, file_path: Path) -> None:
        """Store DataFrame as Parquet file."""
        def _write_parquet():
            table = pa.Table.from_pandas(df)
            pq.write_table(
                table,
                file_path,
                compression=self.config.compression,
                row_group_size=self.config.row_group_size
            )

        await asyncio.get_event_loop().run_in_executor(None, _write_parquet)

    async def _load_parquet(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Load DataFrame from Parquet file."""
        if not file_path.exists():
            return None

        def _read_parquet():
            return pq.read_table(file_path).to_pandas()

        return await asyncio.get_event_loop().run_in_executor(None, _read_parquet)

    async def _store_version_metadata(
        self,
        symbol: str,
        timeframe: str,
        bars: List[Bar],
        version: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Store version metadata."""
        version_info = DataVersion(
            version=version,
            timestamp=datetime.now(timezone.utc),
            checksum=self._generate_version(bars),
            record_count=len(bars),
            schema_hash=self._get_schema_hash(bars),
            metadata=metadata
        )

        # Store as JSON file
        metadata_path = self._get_file_path(symbol, timeframe, bars[0].timestamp.date(), version).with_suffix('.metadata.json')
        await self._write_json(version_info.model_dump(), metadata_path)

    async def _store_feature_version_metadata(
        self,
        symbol: str,
        feature_name: str,
        version: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Store feature version metadata."""
        version_info = DataVersion(
            version=version,
            timestamp=datetime.now(timezone.utc),
            checksum=version,  # Version is already a hash
            record_count=0,  # Will be updated when loaded
            schema_hash="",
            metadata=metadata
        )

        # Store as JSON file
        metadata_path = self._get_feature_file_path(symbol, feature_name, date.today(), version).with_suffix('.metadata.json')
        await self._write_json(version_info.model_dump(), metadata_path)

    def _get_schema_hash(self, bars: List[Bar]) -> str:
        """Get hash of bar schema."""
        if not bars:
            return ""
        schema = list(bars[0].model_dump().keys())
        return hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()[:12]

    async def _write_json(self, data: Dict[str, Any], file_path: Path) -> None:
        """Write data as JSON file."""
        def _write():
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)

        await asyncio.get_event_loop().run_in_executor(None, _write)

    async def _find_data_files(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        version: Optional[str] = None
    ) -> List[Path]:
        """Find data files matching criteria."""
        symbol_path = self.base_path / symbol / timeframe

        if not symbol_path.exists():
            return []

        files = []
        for date_path in symbol_path.iterdir():
            if date_path.is_dir() and date_path.name.startswith('20'):  # Date directory
                date_val = date.fromisoformat(date_path.name)

                # Check date range
                if start_date and date_val < start_date:
                    continue
                if end_date and date_val > end_date:
                    continue

                # Find parquet files
                for file_path in date_path.glob("*.parquet"):
                    if version:
                        if f"{version}.parquet" in file_path.name:
                            files.append(file_path)
                    else:
                        files.append(file_path)

        return sorted(files)

    async def _find_feature_files(
        self,
        symbol: str,
        feature_name: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        version: Optional[str] = None
    ) -> List[Path]:
        """Find feature files matching criteria."""
        feature_path = self.base_path / "features" / symbol / feature_name

        if not feature_path.exists():
            return []

        files = []
        for date_path in feature_path.iterdir():
            if date_path.is_dir() and date_path.name.startswith('20'):  # Date directory
                date_val = date.fromisoformat(date_path.name)

                # Check date range
                if start_date and date_val < start_date:
                    continue
                if end_date and date_val > end_date:
                    continue

                # Find parquet files
                for file_path in date_path.glob("*.parquet"):
                    if version:
                        if f"{version}.parquet" in file_path.name:
                            files.append(file_path)
                    else:
                        files.append(file_path)

        return sorted(files)

    async def _update_duckdb_index(
        self,
        symbol: str,
        timeframe: str,
        date_val: date,
        file_path: Path,
        version: str
    ) -> None:
        """Update DuckDB index with file information."""
        if not self.duckdb_conn:
            return

        def _update():
            self.duckdb_conn.execute("""
                INSERT OR REPLACE INTO file_index
                (symbol, timeframe, date, file_path, version, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [symbol, timeframe, date_val, str(file_path), version, datetime.now(timezone.utc)])

        await asyncio.get_event_loop().run_in_executor(None, _update)

    async def cleanup_old_versions(self, max_versions: Optional[int] = None) -> None:
        """Clean up old versions to save space."""
        max_vers = max_versions or self.config.max_versions

        # This is a simplified implementation
        # In practice, you'd want more sophisticated cleanup logic
        logger.info(f"Cleaning up versions, keeping last {max_vers}")

    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about stored data."""
        info = {
            "base_path": str(self.base_path),
            "total_size_mb": 0,
            "symbol_count": 0,
            "timeframe_count": 0,
            "duckdb_enabled": self.config.enable_duckdb
        }

        if self.base_path.exists():
            total_size = 0
            for file_path in self.base_path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            info["total_size_mb"] = total_size / (1024 * 1024)

            # Count symbols and timeframes
            symbols = [d.name for d in self.base_path.iterdir() if d.is_dir() and d.name != "features"]
            info["symbol_count"] = len(symbols)

        return info