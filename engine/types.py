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
        use_enum_values=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: str(v)
        }
    )
    
    timestamp: datetime = Field(..., description="Bar timestamp in UTC")
    open: Decimal = Field(..., ge=0, description="Open price")
    high: Decimal = Field(..., ge=0, description="High price")
    low: Decimal = Field(..., ge=0, description="Low price")
    close: Decimal = Field(..., ge=0, description="Close price")
    volume: int = Field(..., ge=0, description="Volume")
    
    @field_validator('high')
    @classmethod
    def validate_high(cls, v: Decimal, values: Dict[str, Any]) -> Decimal:
        """Ensure high is the maximum price."""
        if 'low' in values and 'open' in values and 'close' in values:
            low = values['low']
            open_price = values['open']
            close = values['close']
            max_price = max(open_price, close, low)
            if v < max_price:
                raise ValueError(f"High must be >= max(open, close, low), got {v} < {max_price}")
        return v
    
    @field_validator('low')
    @classmethod
    def validate_low(cls, v: Decimal, values: Dict[str, Any]) -> Decimal:
        """Ensure low is the minimum price."""
        if 'high' in values and 'open' in values and 'close' in values:
            high = values['high']
            open_price = values['open']
            close = values['close']
            min_price = min(open_price, close, high)
            if v > min_price:
                raise ValueError(f"Low must be <= min(open, close, high), got {v} > {min_price}")
        return v


class SwingPoint(BaseModel):
    """Represents a swing high or low point."""
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: str(v)
        }
    )
    
    bar_index: int = Field(..., ge=0, description="Index of the bar in the sequence")
    timestamp: datetime = Field(..., description="Timestamp of the swing point")
    price: Decimal = Field(..., gt=0, description="Price level of the swing")
    swing_type: SwingType = Field(..., description="Type of swing point")
    strength: int = Field(..., ge=1, le=10, description="Swing strength (1-10)")
    
    @field_validator('strength')
    @classmethod
    def validate_strength(cls, v: int) -> int:
        """Ensure strength is within valid range."""
        if not 1 <= v <= 10:
            raise ValueError(f"Strength must be between 1 and 10, got {v}")
        return v


class MSS(BaseModel):
    """Market Structure Shift (MSS) - indicates trend change."""
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: str(v)
        }
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
        use_enum_values=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: str(v)
        }
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
        use_enum_values=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: str(v)
        }
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
        use_enum_values=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: str(v)
        }
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
        use_enum_values=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: str(v)
        }
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
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: Decimal) -> Decimal:
        """Ensure confidence is within valid range."""
        if not 0 <= v <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {v}")
        return v


class OrderIntent(BaseModel):
    """Order intent before submission."""
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: str(v)
        }
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
        use_enum_values=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: str(v)
        }
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
def serialize_to_json(obj: BaseModel) -> str:
    """Serialize model to JSON with deterministic ordering."""
    return obj.model_dump_json(exclude_none=True, indent=None, sort_keys=True)


def deserialize_from_json(json_str: str, model_type: type[BaseModel]) -> BaseModel:
    """Deserialize JSON string to model."""
    return model_type.model_validate_json(json_str)