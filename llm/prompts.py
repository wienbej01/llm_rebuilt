"""
LLM prompts for PSE-LLM trading system.
Prompt pack for different LLM tasks and roles.
"""

from __future__ import annotations

from typing import Dict, Any
import json

from llm.schemas import LLMTask, MarketContext, RiskPolicy


class PromptPack:
    """Collection of LLM prompts for different tasks."""

    def __init__(self):
        """Initialize prompt pack."""
        self.global_system_prompt = self._get_global_system_prompt()
        self.task_prompts = {
            LLMTask.VALIDATE_AND_RANK: self._get_validator_prompt(),
            LLMTask.PROPOSE_HOLISTIC: self._get_holistic_proposer_prompt(),
            LLMTask.RISK_VETO_OR_SCALE: self._get_risk_officer_prompt()
        }

    def get_prompt(self, task: LLMTask) -> str:
        """
        Get prompt for a specific task.

        Args:
            task: LLM task type

        Returns:
            Prompt string
        """
        return self.task_prompts.get(task, "")

    def get_system_prompt(self) -> str:
        """
        Get global system prompt.

        Returns:
            System prompt string
        """
        return self.global_system_prompt

    def _get_global_system_prompt(self) -> str:
        """Get global system prompt for all roles."""
        return """You are the Trading LLM inside a quant stack. You must:
- Obey the JSON schema exactly. No free text.
- Use only fields given in input. No speculation.
- Enforce risk policy: reject violations of caps or missing fields.
- Keep reasons terse (≤120 tokens).
- Tactics allowed: DAS 2.1, DAS 2.2, TCEA-FVG-MR, TCEA-SPMF, TCEA-MTE.
- Every setup must include: tactic, side, entry, sl, tp1, rationale tags, and evidence{mqs, frs, tcc, sweep_meta?}."""

    def _get_validator_prompt(self) -> str:
        """Get validator prompt for validate_and_rank task."""
        return """TASK: validate_and_rank
CONTEXT: market_context, risk_policy, setups (source:"PSE")
CHECKLIST:
- DAS 2.1: retest at FVG midpoint; valid FVG width; not stale; SL=opposite FVG edge ± buffer; TP1 from kernel; cite MQS/FRS.
- DAS 2.2: all DAS 2.1 rules + post-sweep tag with priority tier.
- TCEA-FVG-MR: POI missed; reaction ≥ threshold; 1m MSS+pullback; entry on 1m PB breakout; bounded micro SL.
- TCEA-SPMF: TCC active; retracement bounds; micro-FVGs; 5m close breaks PB structure; SL at PB extreme.
- TCEA-MTE: TCC High; thrust bar breaks SR or new MSS with high MQS; entry at first 1m of next 5m; SL at trigger bar extreme.
DECIDE: verdict, priority[0..1], risk_flags, optional numeric edits (entry/sl/tp1).
OUTPUT: {"validations":[...], "holistic_proposals":[]}"""

    def _get_holistic_proposer_prompt(self) -> str:
        """Get holistic proposer prompt for propose_holistic task."""
        return """TASK: propose_holistic
CONTEXT: market_context, risk_policy, observed_features{mqs, frs, sweep_meta, 1m_clues, retracement, thrust_break}
RULES:
- Propose ≤2 setups only from {DAS 2.1, DAS 2.2, TCEA-FVG-MR, TCEA-SPMF, TCEA-MTE}.
- Include: tactic, side, entry, sl, tp1, rationale tags, evidence{mqs, frs, tcc, sweep_meta?}.
- Enforce universal SL cap; tp1 ≥ entry ± k*SL (directional).
OUTPUT: {"validations":[], "holistic_proposals":[...]}"""

    def _get_risk_officer_prompt(self) -> str:
        """Get risk officer prompt for risk_veto_or_scale task."""
        return """TASK: risk_veto_or_scale
INPUT: risk_policy{max_r, sl_cap_points, daily_dd_remaining, max_concurrent_risk},
       candidate_orders[OrderIntent], context{today_trades_count, exposure_risk, spread_estimate}
ACTIONS:
- Veto if SL points > sl_cap_points or news blackout/spread issues.
- Scale down proportionally if concurrent risk would exceed limit.
- If daily_dd_remaining < projected SL loss, veto lowest-priority orders.
OUTPUT: [{"proposal_id":"...","veto":bool,"scale":0..1,"reason":"..."}]"""


class PromptBuilder:
    """Builder for creating formatted prompts with context."""

    def __init__(self, prompt_pack: PromptPack):
        """
        Initialize prompt builder.

        Args:
            prompt_pack: Prompt pack to use
        """
        self.prompt_pack = prompt_pack

    def build_validator_prompt(
        self,
        market_context: MarketContext,
        risk_policy: RiskPolicy,
        setups: list
    ) -> str:
        """
        Build validator prompt with context.

        Args:
            market_context: Market context
            risk_policy: Risk policy
            setups: List of setup proposals

        Returns:
            Formatted prompt
        """
        system_prompt = self.prompt_pack.get_system_prompt()
        task_prompt = self.prompt_pack.get_prompt(LLMTask.VALIDATE_AND_RANK)

        # Format context
        context = {
            "market_context": {
                "symbol": market_context.symbol,
                "tcc": market_context.tcc,
                "mcs": market_context.mcs,
                "session": market_context.session,
                "spread_estimate": market_context.spread_estimate,
                "latency_estimate_ms": market_context.latency_estimate_ms
            },
            "risk_policy": {
                "max_r": risk_policy.max_r,
                "daily_dd": risk_policy.daily_dd,
                "sl_cap_points": risk_policy.sl_cap_points,
                "max_trades_per_day": risk_policy.max_trades_per_day
            },
            "setups": [self._serialize_setup(setup) for setup in setups]
        }

        return f"{system_prompt}\n\n{task_prompt}\n\nCONTEXT:\n{json.dumps(context, indent=2)}"

    def build_holistic_proposer_prompt(
        self,
        market_context: MarketContext,
        risk_policy: RiskPolicy,
        observed_features: Dict[str, Any]
    ) -> str:
        """
        Build holistic proposer prompt with context.

        Args:
            market_context: Market context
            risk_policy: Risk policy
            observed_features: Observed features

        Returns:
            Formatted prompt
        """
        system_prompt = self.prompt_pack.get_system_prompt()
        task_prompt = self.prompt_pack.get_prompt(LLMTask.PROPOSE_HOLISTIC)

        # Format context
        context = {
            "market_context": {
                "symbol": market_context.symbol,
                "tcc": market_context.tcc,
                "mcs": market_context.mcs,
                "session": market_context.session,
                "spread_estimate": market_context.spread_estimate,
                "latency_estimate_ms": market_context.latency_estimate_ms
            },
            "risk_policy": {
                "max_r": risk_policy.max_r,
                "daily_dd": risk_policy.daily_dd,
                "sl_cap_points": risk_policy.sl_cap_points,
                "max_trades_per_day": risk_policy.max_trades_per_day
            },
            "observed_features": observed_features
        }

        return f"{system_prompt}\n\n{task_prompt}\n\nCONTEXT:\n{json.dumps(context, indent=2)}"

    def build_risk_officer_prompt(
        self,
        risk_policy: RiskPolicy,
        candidate_orders: list,
        context: Dict[str, Any]
    ) -> str:
        """
        Build risk officer prompt with context.

        Args:
            risk_policy: Risk policy
            candidate_orders: List of candidate orders
            context: Risk context

        Returns:
            Formatted prompt
        """
        system_prompt = self.prompt_pack.get_system_prompt()
        task_prompt = self.prompt_pack.get_prompt(LLMTask.RISK_VETO_OR_SCALE)

        # Format context
        formatted_orders = []
        for order in candidate_orders:
            formatted_orders.append({
                "proposal_id": order.proposal_id,
                "symbol": order.symbol,
                "side": order.side.value,
                "quantity": order.quantity,
                "entry_price": order.entry_price,
                "stop_loss": order.stop_loss,
                "estimated_risk": order.estimated_risk
            })

        context_data = {
            "risk_policy": {
                "max_r": risk_policy.max_r,
                "daily_dd": risk_policy.daily_dd,
                "sl_cap_points": risk_policy.sl_cap_points,
                "max_trades_per_day": risk_policy.max_trades_per_day
            },
            "candidate_orders": formatted_orders,
            "context": context
        }

        return f"{system_prompt}\n\n{task_prompt}\n\nINPUT:\n{json.dumps(context_data, indent=2)}"

    def _serialize_setup(self, setup) -> Dict[str, Any]:
        """
        Serialize setup proposal for prompt.

        Args:
            setup: Setup proposal

        Returns:
            Serialized setup
        """
        return {
            "id": str(setup.id),
            "symbol": setup.symbol,
            "setup_type": setup.setup_type.value,
            "side": setup.side.value,
            "entry_price": float(setup.entry_price),
            "stop_loss": float(setup.stop_loss),
            "take_profit": float(setup.take_profit),
            "risk_reward_ratio": float(setup.risk_reward_ratio),
            "confidence": float(setup.confidence),
            "evidence": getattr(setup, 'evidence', {})
        }


class PromptTemplates:
    """Template prompts for different scenarios."""

    @staticmethod
    def get_tactic_explanation(tactic: str) -> str:
        """
        Get explanation for a specific tactic.

        Args:
            tactic: Tactic name

        Returns:
            Tactic explanation
        """
        explanations = {
            "DAS 2.1": "MSS + FVG POI Retest: Fresh MSS creates FVG imbalance; shallow retest to POI (midpoint) aligns with structure. Entry at midpoint, SL at opposite FVG edge.",
            "DAS 2.2": "Liquidity Sweep & Reversal: Significant sweep (PDH/PDL > ONH/ONL > SH/SL) precedes reversal MSS; FVG + POI retest gains evidentiary weight.",
            "TCEA-FVG-MR": "Missed FVG → Mean Reversion: FVG midpoint never filled but price reacted strongly. With 1m MSS+pullback confirmation, re-enter on local breakout.",
            "TCEA-SPMF": "Smart Pullback & Micro-FVG: In strong trend, impulse → rhythmic pullback with micro-FVGs; breakout of pullback structure offers continuation.",
            "TCEA-MTE": "Momentum Thrust Entry: With high trend conviction, a thrust bar/MSS that breaks SR offers momentum continuation."
        }

        return explanations.get(tactic, f"Tactic: {tactic}")

    @staticmethod
    def get_risk_guidelines() -> str:
        """Get risk guidelines for LLM."""
        return """Risk Guidelines:
- Universal SL cap: Never exceed maximum SL points
- Position sizing: Calculate based on account risk and stop loss distance
- Daily limits: Respect maximum daily drawdown and trade count
- Spread guard: Skip if spread exceeds threshold at decision time
- Cool-off: Enforce cooling period after losses
- Correlation: Consider correlation with existing positions"""

    @staticmethod
    def get_quality_thresholds() -> str:
        """Get quality thresholds for LLM."""
        return """Quality Thresholds:
- MQS (Market Quality Score): Minimum 6.0, prefer 7.0+
- FRS (Formation Reliability Score): Minimum 6.0, prefer 7.0+
- Confidence: Minimum 0.7 for execution
- Volume confirmation: Required for most setups
- Multiple confirmations: Required for high-risk setups"""

    @staticmethod
    def get_market_context_template() -> str:
        """Get market context template."""
        return """Market Context Template:
{
  "symbol": "ES",
  "tcc": "High",
  "mcs": "TrendUp",
  "session": "RTH",
  "spread_estimate": 0.25,
  "latency_estimate_ms": 50
}"""

    @staticmethod
    def get_setup_template() -> str:
        """Get setup template."""
        return """Setup Template:
{
  "tactic": "DAS 2.1",
  "side": "BUY",
  "entry": 5000.00,
  "sl": 4990.00,
  "tp1": 5015.00,
  "evidence": {
    "mqs": 7.5,
    "frs": 7.0,
    "tcc": "High",
    "swing_strength": 8,
    "fvg_width": 0.002,
    "retest_strength": 0.8
  }
}"""

    @staticmethod
    def get_validation_template() -> str:
        """Get validation template."""
        return """Validation Template:
{
  "proposal_id": "uuid",
  "verdict": "approve",
  "priority": 0.8,
  "reason": "Strong FVG with high MQS and proper retest",
  "risk_flags": {
    "sl_cap_violation": false,
    "news_blackout": false,
    "spread_too_wide": false,
    "too_many_trades_today": false
  },
  "edits": {
    "entry": null,
    "sl": null,
    "tp1": null
  }
}"""

    @staticmethod
    def get_risk_decision_template() -> str:
        """Get risk decision template."""
        return """Risk Decision Template:
{
  "proposal_id": "uuid",
  "veto": false,
  "scale": 1.0,
  "reason": "Within risk limits and correlation guidelines"
}"""


# Global prompt pack instance
prompt_pack = PromptPack()
prompt_builder = PromptBuilder(prompt_pack)


def get_prompt_pack() -> PromptPack:
    """Get the global prompt pack instance."""
    return prompt_pack


def get_prompt_builder() -> PromptBuilder:
    """Get the global prompt builder instance."""
    return prompt_builder