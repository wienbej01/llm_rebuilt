"""
Domain types for the Python Strategy Engine (PSE).
All models use Pydantic v2 with strict typing and deterministic JSON serialization.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import List, Optional, Dict, Any, Union
from uuid import UUID, uuid4
import json

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator


class Side(str, Enum):
    """Trade side enumeration."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type enumeration."""
    MARKET = "MKT"
    LIMIT = "LMT"
    STOP = "STP"
    STOP_LIMIT = "STP_LMT"


class OrderStatus(str, Enum):
    """Order status enumeration."""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class TimeInForce(str, Enum):
    """Time in force enumeration."""
    DAY = "DAY"
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"


class SwingType(str, Enum):
    """Swing point type enumeration."""
    SWING_HIGH = "SWING_HIGH"
    SWING_LOW = "SWING_LOW"


class MSSDirection(str, Enum):
    """Market Structure Shift direction."""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"


class FVGType(str, Enum):
    """Fair Value Gap type."""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"


class SetupType(str, Enum):
    """Trade setup type enumeration."""
    BREAK_RETEST = "BREAK_RETEST"
    ORDER_BLOCK = "ORDER_BLOCK"
    FVG = "FVG"
    LIQUIDITY_SWEEP = "LIQUIDITY_SWEEP"
    CHANGE_OF_CHARACTER = "CHANGE_OF_CHARACTER"


class Bar(BaseModel):
    """Represents a single price bar/candle."""
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True
    )
    
    timestamp: datetime = Field(..., description="Bar timestamp in UTC")
    open: Decimal = Field(..., ge=0, description="Open price")
    high: Decimal = Field(..., ge=0, description="High price")
    low: Decimal = Field(..., ge=0, description="Low price")
    close: Decimal = Field(..., ge=0, description="Close price")
    volume: int = Field(..., ge=0, description="Volume")
    
    @model_validator(mode='after')
    def validate_prices(self) -> 'Bar':
        """Ensure price relationships are valid."""
        if self.high < max(self.open, self.close, self.low):
            raise ValueError(f"High must be >= max(open, close, low), got {self.high} < {max(self.open, self.close, self.low)}")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError(f"Low must be <= min(open, close, high), got {self.low} > {min(self.open, self.close, self.high)}")
        return self


class SwingPoint(BaseModel):
    """Represents a swing high or low point."""
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True
    )
    
    bar_index: int = Field(..., ge=0, description="Index of the bar in the sequence")
    timestamp: datetime = Field(..., description="Timestamp of the swing point")
    price: Decimal = Field(..., gt=0, description="Price level of the swing")
    swing_type: SwingType = Field(..., description="Type of swing point")
    strength: int = Field(..., ge=1, le=10, description="Swing strength (1-10)")


class MSS(BaseModel):
    """Market Structure Shift (MSS) - indicates trend change."""
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True
    )
    
    start_bar: int = Field(..., ge=0, description="Starting bar index")
    end_bar: int = Field(..., ge=0, description="Ending bar index")
    direction: MSSDirection = Field(..., description="Direction of the shift")
    break_price: Decimal = Field(..., gt=0, description="Price level where structure broke")
    confirmation_price: Decimal = Field(..., gt=0, description="Price confirming the shift")
    is_valid: bool = Field(True, description="Whether this MSS is still valid")
    
    @model_validator(mode='after')
    def validate_bar_order(self) -> 'MSS':
        """Ensure start_bar <= end_bar."""
        if self.start_bar > self.end_bar:
            raise ValueError(f"start_bar ({self.start_bar}) must be <= end_bar ({self.end_bar})")
        return self


class FVG(BaseModel):
    """Fair Value Gap - imbalance in price action."""
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True
    )
    
    start_bar: int = Field(..., ge=0, description="Starting bar index")
    end_bar: int = Field(..., ge=0, description="Ending bar index")
    fvg_type: FVGType = Field(..., description="Type of FVG")
    top: Decimal = Field(..., gt=0, description="Top boundary of FVG")
    bottom: Decimal = Field(..., gt=0, description="Bottom boundary of FVG")
    is_filled: bool = Field(False, description="Whether FVG has been filled")
    fill_bar: Optional[int] = Field(None, ge=0, description="Bar index when filled")
    
    @model_validator(mode='after')
    def validate_boundaries(self) -> 'FVG':
        """Ensure top > bottom."""
        if self.top <= self.bottom:
            raise ValueError(f"top ({self.top}) must be > bottom ({self.bottom})")
        return self
    
    @model_validator(mode='after')
    def validate_bar_order(self) -> 'FVG':
        """Ensure start_bar <= end_bar."""
        if self.start_bar > self.end_bar:
            raise ValueError(f"start_bar ({self.start_bar}) must be <= end_bar ({self.end_bar})")
        return self


class TCC(BaseModel):
    """Time Cycle Completion - market cycle analysis."""
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True
    )
    
    start_time: datetime = Field(..., description="Cycle start time")
    end_time: datetime = Field(..., description="Cycle end time")
    cycle_length: int = Field(..., gt=0, description="Length of cycle in bars")
    cycle_type: str = Field(..., description="Type of cycle (e.g., 'impulse', 'corrective')")
    strength: int = Field(..., ge=1, le=10, description="Cycle strength (1-10)")
    
    @model_validator(mode='after')
    def validate_time_order(self) -> 'TCC':
        """Ensure start_time <= end_time."""
        if self.start_time > self.end_time:
            raise ValueError(f"start_time must be <= end_time")
        return self


class MCS(BaseModel):
    """Market Cycle Structure - higher timeframe context."""
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True
    )
    
    timeframe: str = Field(..., description="Higher timeframe")
    structure_type: str = Field(..., description="Type of structure (e.g., 'uptrend', 'downtrend', 'range')")
    bias: str = Field(..., description="Market bias (bullish, bearish, neutral)")
    key_levels: List[Decimal] = Field(..., description="Key support/resistance levels")
    last_updated: datetime = Field(..., description="Last update time")


class SetupProposal(BaseModel):
    """Trade setup proposal with evidence fields."""
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True
    )
    
    id: UUID = Field(default_factory=uuid4, description="Unique identifier")
    symbol: str = Field(..., description="Trading symbol")
    setup_type: SetupType = Field(..., description="Type of setup")
    side: Side = Field(..., description="Trade direction")
    entry_price: Decimal = Field(..., gt=0, description="Proposed entry price")
    stop_loss: Decimal = Field(..., gt=0, description="Stop loss price")
    take_profit: Decimal = Field(..., gt=0, description="Take profit price")
    risk_reward_ratio: Decimal = Field(..., gt=0, description="Risk:Reward ratio")
    confidence: Decimal = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    
    # Evidence fields
    swing_points: List[SwingPoint] = Field(..., description="Relevant swing points")
    mss_list: List[MSS] = Field(..., description="Market structure shifts")
    fvgs: List[FVG] = Field(..., description="Fair value gaps")
    tcc_context: Optional[TCC] = Field(None, description="Time cycle context")
    mcs_context: Optional[MCS] = Field(None, description="Market cycle structure")
    
    # Additional evidence
    volume_analysis: Dict[str, Any] = Field(..., description="Volume-based evidence")
    order_flow: Dict[str, Any] = Field(..., description="Order flow evidence")

    # Detector and risk assessment data
    evidence: Optional[Dict[str, Any]] = Field(None, description="Detector-specific evidence for the setup.")
    risk_assessment: Optional[Dict[str, Any]] = Field(None, description="Risk assessment results for the setup.")

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    @model_validator(mode='after')
    def validate_prices(self) -> 'SetupProposal':
        """Validate price relationships."""
        if self.side == Side.BUY:
            if self.entry_price <= self.stop_loss:
                raise ValueError(f"For BUY, entry_price ({self.entry_price}) must be > stop_loss ({self.stop_loss})")
            if self.take_profit <= self.entry_price:
                raise ValueError(f"For BUY, take_profit ({self.take_profit}) must be > entry_price ({self.entry_price})")
        else:  # SELL
            if self.entry_price >= self.stop_loss:
                raise ValueError(f"For SELL, entry_price ({self.entry_price}) must be < stop_loss ({self.stop_loss})")
            if self.take_profit >= self.entry_price:
                raise ValueError(f"For SELL, take_profit ({self.take_profit}) must be < entry_price ({self.entry_price})")
        return self


class OrderIntent(BaseModel):
    """Order intent before submission."""
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True
    )
    
    id: UUID = Field(default_factory=uuid4, description="Unique identifier")
    symbol: str = Field(..., description="Trading symbol")
    side: Side = Field(..., description="Order side")
    order_type: OrderType = Field(..., description="Order type")
    quantity: int = Field(..., gt=0, description="Order quantity")
    price: Optional[Decimal] = Field(None, gt=0, description="Order price (for limit/stop orders)")
    stop_price: Optional[Decimal] = Field(None, gt=0, description="Stop price (for stop orders)")
    time_in_force: TimeInForce = Field(TimeInForce.DAY, description="Time in force")
    outside_rth: bool = Field(False, description="Allow trading outside regular hours")
    setup_id: UUID = Field(..., description="Associated setup proposal ID")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ExecutionReport(BaseModel):
    """Execution report for filled orders."""
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True
    )
    
    id: UUID = Field(default_factory=uuid4, description="Unique identifier")
    order_id: UUID = Field(..., description="Original order ID")
    symbol: str = Field(..., description="Trading symbol")
    side: Side = Field(..., description="Order side")
    quantity: int = Field(..., gt=0, description="Filled quantity")
    price: Decimal = Field(..., gt=0, description="Fill price")
    commission: Decimal = Field(default=Decimal('0'), ge=0, description="Commission")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: OrderStatus = Field(OrderStatus.FILLED, description="Execution status")
    
    @field_validator('commission')
    @classmethod
    def validate_commission(cls, v: Decimal) -> Decimal:
        """Ensure commission is non-negative."""
        if v < 0:
            raise ValueError(f"Commission must be non-negative, got {v}")
        return v


# Utility functions for deterministic serialization
def decimal_serializer(obj):
    """Custom serializer for Decimal objects."""
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def serialize_to_json(obj: BaseModel) -> str:
    """Serialize model to JSON with deterministic ordering."""
    # Convert to dict first, then to JSON with sorted keys
    data = obj.model_dump(exclude_none=True)
    return json.dumps(data, sort_keys=True, separators=(',', ':'), default=decimal_serializer)


def deserialize_from_json(json_str: str, model_type: type[BaseModel]) -> BaseModel:
    """Deserialize JSON string to model."""
    return model_type.model_validate_json(json_str)