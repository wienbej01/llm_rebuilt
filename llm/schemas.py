"""
LLM schemas for PSE-LLM trading system.
Pydantic models for LLM input/output validation.
"""

from __future__ import annotations
from decimal import Decimal

from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import uuid

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

from engine.types import SetupProposal, Side, SetupType


class LLMTask(str, Enum):
    """LLM task types."""
    VALIDATE_AND_RANK = "validate_and_rank"
    PROPOSE_HOLISTIC = "propose_holistic"
    RISK_VETO_OR_SCALE = "risk_veto_or_scale"


class MarketContext(BaseModel):
    """Market context for LLM."""
    symbol: str = Field(..., description="Trading symbol")
    tcc: str = Field(..., description="Time Cycle Completion state")
    mcs: str = Field(..., description="Market Cycle Structure state")
    session: str = Field(..., description="Trading session (RTH/ETH)")
    spread_estimate: Decimal = Field(..., ge=0, description="Estimated spread")
    latency_estimate_ms: float = Field(..., ge=0, description="Estimated latency in ms")

    model_config = ConfigDict(frozen=True)


class RiskPolicy(BaseModel):
    """Risk policy for LLM."""
    max_r: Decimal = Field(..., gt=0, description="Max R per trade")
    daily_dd: Decimal = Field(..., gt=0, le=1, description="Max daily drawdown")
    sl_cap_points: Decimal = Field(..., gt=0, description="SL cap in points")
    max_trades_per_day: int = Field(..., ge=0, description="Max trades per day")

    model_config = ConfigDict(frozen=True)


class RiskFlags(BaseModel):
    """Risk flags for validation."""
    sl_cap_violation: bool = Field(default=False, description="SL cap violation")
    news_blackout: bool = Field(default=False, description="News blackout")
    spread_too_wide: bool = Field(default=False, description="Spread too wide")
    too_many_trades_today: bool = Field(default=False, description="Too many trades today")

    model_config = ConfigDict(frozen=True)


class SetupEdit(BaseModel):
    """Setup edit suggestions."""
    entry: Optional[Decimal] = Field(None, description="Entry price edit")
    sl: Optional[Decimal] = Field(None, description="Stop loss edit")
    tp1: Optional[Decimal] = Field(None, description="Take profit 1 edit")

    model_config = ConfigDict(frozen=True)


class Validation(BaseModel):
    """Setup validation result."""
    proposal_id: str = Field(..., description="Setup proposal ID")
    verdict: str = Field(..., description="Validation verdict (approve/reject/revise)")
    priority: float = Field(..., ge=0, le=1, description="Priority score")
    reason: str = Field(..., max_length=120, description="Validation reason")
    risk_flags: RiskFlags = Field(default_factory=RiskFlags, description="Risk flags")
    edits: SetupEdit = Field(default_factory=SetupEdit, description="Suggested edits")

    model_config = ConfigDict(frozen=True)

    @field_validator('verdict')
    @classmethod
    def validate_verdict(cls, v: str) -> str:
        """Validate verdict value."""
        allowed = {"approve", "reject", "revise"}
        if v not in allowed:
            raise ValueError(f"Verdict must be one of {allowed}")
        return v


class RiskDecision(BaseModel):
    """Risk decision for setup."""
    proposal_id: str = Field(..., description="Setup proposal ID")
    veto: bool = Field(default=False, description="Veto decision")
    scale: float = Field(..., ge=0, le=1, description="Scale factor")
    reason: str = Field(..., max_length=120, description="Decision reason")

    model_config = ConfigDict(frozen=True)


class LLMInput(BaseModel):
    """LLM input schema."""
    session: str = Field(..., description="Session identifier")
    market_context: MarketContext = Field(..., description="Market context")
    setups: List[SetupProposal] = Field(..., description="Setup proposals")
    risk_policy: RiskPolicy = Field(..., description="Risk policy")
    ask: LLMTask = Field(..., description="LLM task")

    model_config = ConfigDict(frozen=True)

    @field_validator('setups')
    @classmethod
    def validate_setups(cls, v: List[SetupProposal]) -> List[SetupProposal]:
        """Validate setup proposals."""
        if len(v) > 10:
            raise ValueError("Maximum 10 setups allowed")
        return v


class LLMOutput(BaseModel):
    """LLM output schema."""
    validations: List[Validation] = Field(default_factory=list, description="Setup validations")
    holistic_proposals: List[SetupProposal] = Field(default_factory=list, description="Holistic proposals")
    risk_decisions: List[RiskDecision] = Field(default_factory=list, description="Risk decisions")

    model_config = ConfigDict(frozen=True)

    @model_validator(mode='after')
    def validate_output_consistency(self) -> 'LLMOutput':
        """Validate output consistency."""
        # For validate_and_rank, only validations should be present
        if self.validations and (self.holistic_proposals or self.risk_decisions):
            raise ValueError("For validate_and_rank, only validations should be present")

        # For propose_holistic, only holistic_proposals should be present
        if self.holistic_proposals and (self.validations or self.risk_decisions):
            raise ValueError("For propose_holistic, only holistic_proposals should be present")

        # For risk_veto_or_scale, only risk_decisions should be present
        if self.risk_decisions and (self.validations or self.holistic_proposals):
            raise ValueError("For risk_veto_or_scale, only risk_decisions should be present")

        return self


class OrderIntent(BaseModel):
    """Order intent for risk assessment."""
    proposal_id: str = Field(..., description="Associated proposal ID")
    symbol: str = Field(..., description="Trading symbol")
    side: Side = Field(..., description="Order side")
    quantity: int = Field(..., gt=0, description="Order quantity")
    entry_price: Decimal = Field(..., gt=0, description="Entry price")
    stop_loss: Decimal = Field(..., gt=0, description="Stop loss price")
    estimated_risk: Decimal = Field(..., ge=0, description="Estimated risk amount")

    model_config = ConfigDict(frozen=True)


class RiskContext(BaseModel):
    """Risk context for LLM."""
    today_trades_count: int = Field(..., ge=0, description="Trades today")
    exposure_risk: Decimal = Field(..., ge=0, description="Current exposure risk")
    spread_estimate: Decimal = Field(..., ge=0, description="Current spread estimate")
    daily_dd_remaining: Decimal = Field(..., ge=0, le=1, description="Daily drawdown remaining")

    model_config = ConfigDict(frozen=True)


class RiskInput(BaseModel):
    """Risk assessment input schema."""
    risk_policy: RiskPolicy = Field(..., description="Risk policy")
    candidate_orders: List[OrderIntent] = Field(..., description="Candidate orders")
    context: RiskContext = Field(..., description="Risk context")

    model_config = ConfigDict(frozen=True)

    @field_validator('candidate_orders')
    @classmethod
    def validate_orders(cls, v: List[OrderIntent]) -> List[OrderIntent]:
        """Validate candidate orders."""
        if len(v) > 10:
            raise ValueError("Maximum 10 orders allowed")
        return v


class ObservedFeatures(BaseModel):
    """Observed market features for holistic proposals."""
    mqs: float = Field(..., ge=0, le=10, description="Market Quality Score")
    frs: float = Field(..., ge=0, le=10, description="Formation Reliability Score")
    sweep_meta: Optional[Dict[str, Any]] = Field(None, description="Sweep metadata")
    clues_1m: List[str] = Field(default_factory=list, description="1-minute clues")
    retracement: float = Field(..., ge=0, description="Retracement level")
    thrust_break: bool = Field(default=False, description="Thrust break detected")

    model_config = ConfigDict(frozen=True)


class HolisticInput(BaseModel):
    """Holistic proposal input schema."""
    session: str = Field(..., description="Session identifier")
    market_context: MarketContext = Field(..., description="Market context")
    observed_features: ObservedFeatures = Field(..., description="Observed features")
    risk_policy: RiskPolicy = Field(..., description="Risk policy")

    model_config = ConfigDict(frozen=True)


class LLMRequestConfig(BaseModel):
    """LLM request configuration."""
    provider: str = Field(..., description="LLM provider")
    model: str = Field(..., description="LLM model")
    temperature: float = Field(default=0.1, ge=0, le=2, description="Temperature")
    max_tokens: Optional[int] = Field(default=1000, ge=1, description="Max tokens")
    timeout: float = Field(default=30.0, ge=1, description="Timeout in seconds")
    retry_attempts: int = Field(default=3, ge=0, description="Retry attempts")

    model_config = ConfigDict(frozen=True)


class LLMResponseMetadata(BaseModel):
    """LLM response metadata."""
    provider: str = Field(..., description="LLM provider")
    model: str = Field(..., description="LLM model")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    response_time_ms: float = Field(..., ge=0, description="Response time in ms")
    tokens_used: int = Field(..., ge=0, description="Tokens used")
    cost_estimate: Decimal = Field(..., ge=0, description="Cost estimate in USD")

    model_config = ConfigDict(frozen=True)


class LLMInteraction(BaseModel):
    """Complete LLM interaction record."""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Request ID")
    config: LLMRequestConfig = Field(..., description="Request configuration")
    input_data: Union[LLMInput, RiskInput, HolisticInput] = Field(..., description="Input data")
    output_data: Optional[Union[LLMOutput, List[RiskDecision]]] = Field(None, description="Output data")
    metadata: LLMResponseMetadata = Field(..., description="Response metadata")
    success: bool = Field(..., description="Success flag")
    error_message: Optional[str] = Field(None, description="Error message if failed")

    model_config = ConfigDict(frozen=True)


class LLMStats(BaseModel):
    """LLM usage statistics."""
    total_requests: int = Field(default=0, ge=0, description="Total requests")
    successful_requests: int = Field(default=0, ge=0, description="Successful requests")
    failed_requests: int = Field(default=0, ge=0, description="Failed requests")
    average_response_time_ms: float = Field(default=0.0, ge=0, description="Average response time")
    total_tokens_used: int = Field(default=0, ge=0, description="Total tokens used")
    total_cost_estimate: Decimal = Field(default=0.0, ge=0, description="Total cost estimate")
    last_request_time: Optional[datetime] = Field(None, description="Last request time")

    model_config = ConfigDict(frozen=True)


# Utility functions for schema validation
def validate_llm_input(data: Dict[str, Any], task: LLMTask) -> LLMInput:
    """
    Validate and create LLM input.

    Args:
        data: Input data dictionary
        task: LLM task type

    Returns:
        Validated LLM input
    """
    data["ask"] = task
    return LLMInput.model_validate(data)


def validate_llm_output(data: Dict[str, Any], task: LLMTask) -> LLMOutput:
    """
    Validate and create LLM output.

    Args:
        data: Output data dictionary
        task: LLM task type

    Returns:
        Validated LLM output
    """
    output = LLMOutput.model_validate(data)

    # Additional task-specific validation
    if task == LLMTask.VALIDATE_AND_RANK and not output.validations:
        raise ValueError("validate_and_rank task requires validations")
    elif task == LLMTask.PROPOSE_HOLISTIC and not output.holistic_proposals:
        raise ValueError("propose_holistic task requires holistic_proposals")
    elif task == LLMTask.RISK_VETO_OR_SCALE and not output.risk_decisions:
        raise ValueError("risk_veto_or_scale task requires risk_decisions")

    return output


def create_market_context(
    symbol: str,
    tcc: str,
    mcs: str,
    session: str,
    spread_estimate: Decimal,
    latency_estimate_ms: float
) -> MarketContext:
    """
    Create market context.

    Args:
        symbol: Trading symbol
        tcc: TCC state
        mcs: MCS state
        session: Trading session
        spread_estimate: Spread estimate
        latency_estimate_ms: Latency estimate

    Returns:
        Market context
    """
    return MarketContext(
        symbol=symbol,
        tcc=tcc,
        mcs=mcs,
        session=session,
        spread_estimate=spread_estimate,
        latency_estimate_ms=latency_estimate_ms
    )


def create_risk_policy(
    max_r: Decimal,
    daily_dd: Decimal,
    sl_cap_points: Decimal,
    max_trades_per_day: int
) -> RiskPolicy:
    """
    Create risk policy.

    Args:
        max_r: Max R per trade
        daily_dd: Daily drawdown
        sl_cap_points: SL cap in points
        max_trades_per_day: Max trades per day

    Returns:
        Risk policy
    """
    return RiskPolicy(
        max_r=max_r,
        daily_dd=daily_dd,
        sl_cap_points=sl_cap_points,
        max_trades_per_day=max_trades_per_day
    )


def create_validation(
    proposal_id: str,
    verdict: str,
    priority: float,
    reason: str,
    risk_flags: Optional[RiskFlags] = None,
    edits: Optional[SetupEdit] = None
) -> Validation:
    """
    Create validation result.

    Args:
        proposal_id: Proposal ID
        verdict: Verdict
        priority: Priority
        reason: Reason
        risk_flags: Risk flags
        edits: Setup edits

    Returns:
        Validation
    """
    return Validation(
        proposal_id=proposal_id,
        verdict=verdict,
        priority=priority,
        reason=reason,
        risk_flags=risk_flags or RiskFlags(),
        edits=edits or SetupEdit()
    )


def create_risk_decision(
    proposal_id: str,
    veto: bool,
    scale: float,
    reason: str
) -> RiskDecision:
    """
    Create risk decision.

    Args:
        proposal_id: Proposal ID
        veto: Veto flag
        scale: Scale factor
        reason: Reason

    Returns:
        Risk decision
    """
    return RiskDecision(
        proposal_id=proposal_id,
        veto=veto,
        scale=scale,
        reason=reason
    )