"""
LLM agents for PSE-LLM trading system.
Different agent roles for various LLM tasks.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
import logging

from llm.gateway import LLMGateway, LLMProvider, LLMModel
from llm.schemas import (
    LLMInput, LLMOutput, Validation, RiskDecision, MarketContext,
    RiskPolicy, LLMTask, OrderIntent, RiskInput, RiskContext,
    ObservedFeatures, HolisticInput, LLMRequestConfig, LLMResponseMetadata
)
from llm.prompts import PromptBuilder, PromptPack
from llm.validators import LLMValidator, LLMValidationError

logger = logging.getLogger(__name__)


class LLMAgent:
    """Base class for LLM agents."""

    def __init__(
        self,
        gateway: LLMGateway,
        prompt_builder: PromptBuilder,
        validator: LLMValidator,
        config: LLMRequestConfig
    ):
        """
        Initialize LLM agent.

        Args:
            gateway: LLM gateway
            prompt_builder: Prompt builder
            validator: LLM validator
            config: Request configuration
        """
        self.gateway = gateway
        self.prompt_builder = prompt_builder
        self.validator = validator
        self.config = config

        # Agent statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time_ms": 0.0,
            "last_request_time": None
        }

        logger.info(f"Initialized {self.__class__.__name__}")

    async def process_request(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Process request (to be implemented by subclasses).

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Response data
        """
        raise NotImplementedError("Subclasses must implement process_request")

    def _update_stats(self, success: bool, response_time_ms: float) -> None:
        """Update agent statistics."""
        self.stats["total_requests"] += 1
        self.stats["last_request_time"] = datetime.now(timezone.utc)

        if success:
            self.stats["successful_requests"] += 1
        else:
            self.stats["failed_requests"] += 1

        # Update average response time
        total_requests = self.stats["total_requests"]
        current_avg = self.stats["average_response_time_ms"]
        self.stats["average_response_time_ms"] = (
            (current_avg * (total_requests - 1) + response_time_ms) / total_requests
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return self.stats.copy()


class ValidatorAgent(LLMAgent):
    """Agent for validating and ranking setup proposals."""

    async def process_request(
        self,
        market_context: MarketContext,
        risk_policy: RiskPolicy,
        setups: List[Dict[str, Any]]
    ) -> LLMOutput:
        """
        Validate and rank setup proposals.

        Args:
            market_context: Market context
            risk_policy: Risk policy
            setups: List of setup proposals

        Returns:
            Validation results
        """
        start_time = datetime.now(timezone.utc)

        try:
            # Build prompt
            prompt = self.prompt_builder.build_validator_prompt(
                market_context, risk_policy, setups
            )

            # Send request to LLM
            response = await self.gateway.send_json_request(
                provider=LLMProvider(self.config.provider),
                model=LLMModel(self.config.model),
                system_prompt=self.prompt_builder.prompt_pack.get_system_prompt(),
                user_prompt=prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout,
                retry_attempts=self.config.retry_attempts
            )

            # Validate and parse response
            response_data = await self.gateway.validate_json_response(response)
            llm_output = self.validator.validate_output(response_data, LLMTask.VALIDATE_AND_RANK)

            # Update stats
            response_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self._update_stats(True, response_time)

            logger.debug(f"Validator agent processed {len(setups)} setups")
            return llm_output

        except Exception as e:
            response_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self._update_stats(False, response_time)
            logger.error(f"Validator agent failed: {e}")
            raise


class HolisticProposerAgent(LLMAgent):
    """Agent for proposing holistic trade setups."""

    async def process_request(
        self,
        market_context: MarketContext,
        risk_policy: RiskPolicy,
        observed_features: ObservedFeatures
    ) -> LLMOutput:
        """
        Propose holistic trade setups.

        Args:
            market_context: Market context
            risk_policy: Risk policy
            observed_features: Observed market features

        Returns:
            Holistic proposals
        """
        start_time = datetime.now(timezone.utc)

        try:
            # Build prompt
            prompt = self.prompt_builder.build_holistic_proposer_prompt(
                market_context, risk_policy, observed_features.model_dump()
            )

            # Send request to LLM
            response = await self.gateway.send_json_request(
                provider=LLMProvider(self.config.provider),
                model=LLMModel(self.config.model),
                system_prompt=self.prompt_builder.prompt_pack.get_system_prompt(),
                user_prompt=prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout,
                retry_attempts=self.config.retry_attempts
            )

            # Validate and parse response
            response_data = await self.gateway.validate_json_response(response)
            llm_output = self.validator.validate_output(response_data, LLMTask.PROPOSE_HOLISTIC)

            # Update stats
            response_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self._update_stats(True, response_time)

            logger.debug(f"Holistic proposer agent generated {len(llm_output.holistic_proposals)} proposals")
            return llm_output

        except Exception as e:
            response_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self._update_stats(False, response_time)
            logger.error(f"Holistic proposer agent failed: {e}")
            raise


class RiskOfficerAgent(LLMAgent):
    """Agent for risk assessment and decision making."""

    async def process_request(
        self,
        risk_policy: RiskPolicy,
        candidate_orders: List[OrderIntent],
        context: RiskContext
    ) -> List[RiskDecision]:
        """
        Assess risk and make decisions.

        Args:
            risk_policy: Risk policy
            candidate_orders: Candidate orders
            context: Risk context

        Returns:
            Risk decisions
        """
        start_time = datetime.now(timezone.utc)

        try:
            # Build prompt
            prompt = self.prompt_builder.build_risk_officer_prompt(
                risk_policy, candidate_orders, context.model_dump()
            )

            # Send request to LLM
            response = await self.gateway.send_json_request(
                provider=LLMProvider(self.config.provider),
                model=LLMModel(self.config.model),
                system_prompt=self.prompt_builder.prompt_pack.get_system_prompt(),
                user_prompt=prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout,
                retry_attempts=self.config.retry_attempts
            )

            # Validate and parse response
            response_data = await self.gateway.validate_json_response(response)
            llm_output = self.validator.validate_output(response_data, LLMTask.RISK_VETO_OR_SCALE)

            # Update stats
            response_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self._update_stats(True, response_time)

            logger.debug(f"Risk officer agent processed {len(candidate_orders)} orders")
            return llm_output.risk_decisions

        except Exception as e:
            response_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self._update_stats(False, response_time)
            logger.error(f"Risk officer agent failed: {e}")
            raise


class LLMOrchestrator:
    """Orchestrator for coordinating multiple LLM agents."""

    def __init__(
        self,
        gateway: LLMGateway,
        prompt_pack: PromptPack = None,
        validator: LLMValidator = None
    ):
        """
        Initialize LLM orchestrator.

        Args:
            gateway: LLM gateway
            prompt_pack: Prompt pack (optional)
            validator: LLM validator (optional)
        """
        self.gateway = gateway
        self.prompt_pack = prompt_pack or PromptPack()
        self.validator = validator or LLMValidator()
        self.prompt_builder = PromptBuilder(self.prompt_pack)

        # Initialize agents
        self.agents = {}

        # Orchestrator statistics
        self.stats = {
            "total_orchestrations": 0,
            "successful_orchestrations": 0,
            "failed_orchestrations": 0,
            "average_orchestration_time_ms": 0.0,
            "last_orchestration_time": None
        }

        logger.info("Initialized LLM orchestrator")

    def register_agent(
        self,
        name: str,
        agent_class: type[LLMAgent],
        config: LLMRequestConfig
    ) -> None:
        """
        Register an LLM agent.

        Args:
            name: Agent name
            agent_class: Agent class
            config: Request configuration
        """
        agent = agent_class(
            gateway=self.gateway,
            prompt_builder=self.prompt_builder,
            validator=self.validator,
            config=config
        )
        self.agents[name] = agent
        logger.info(f"Registered agent: {name}")

    def get_agent(self, name: str) -> LLMAgent:
        """
        Get registered agent.

        Args:
            name: Agent name

        Returns:
            Agent instance

        Raises:
            ValueError: If agent not found
        """
        if name not in self.agents:
            raise ValueError(f"Agent not found: {name}")
        return self.agents[name]

    async def validate_and_rank_setups(
        self,
        market_context: MarketContext,
        risk_policy: RiskPolicy,
        setups: List[Dict[str, Any]]
    ) -> LLMOutput:
        """
        Validate and rank setup proposals.

        Args:
            market_context: Market context
            risk_policy: Risk policy
            setups: List of setup proposals

        Returns:
            Validation results
        """
        start_time = datetime.now(timezone.utc)

        try:
            validator_agent = self.get_agent("validator")
            result = await validator_agent.process_request(
                market_context, risk_policy, setups
            )

            # Update stats
            orchestration_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self._update_stats(True, orchestration_time)

            return result

        except Exception as e:
            orchestration_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self._update_stats(False, orchestration_time)
            logger.error(f"Validate and rank setups failed: {e}")
            raise

    async def propose_holistic_setups(
        self,
        market_context: MarketContext,
        risk_policy: RiskPolicy,
        observed_features: ObservedFeatures
    ) -> LLMOutput:
        """
        Propose holistic trade setups.

        Args:
            market_context: Market context
            risk_policy: Risk policy
            observed_features: Observed market features

        Returns:
            Holistic proposals
        """
        start_time = datetime.now(timezone.utc)

        try:
            proposer_agent = self.get_agent("holistic_proposer")
            result = await proposer_agent.process_request(
                market_context, risk_policy, observed_features
            )

            # Update stats
            orchestration_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self._update_stats(True, orchestration_time)

            return result

        except Exception as e:
            orchestration_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self._update_stats(False, orchestration_time)
            logger.error(f"Propose holistic setups failed: {e}")
            raise

    async def assess_risk_decisions(
        self,
        risk_policy: RiskPolicy,
        candidate_orders: List[OrderIntent],
        context: RiskContext
    ) -> List[RiskDecision]:
        """
        Assess risk and make decisions.

        Args:
            risk_policy: Risk policy
            candidate_orders: Candidate orders
            context: Risk context

        Returns:
            Risk decisions
        """
        start_time = datetime.now(timezone.utc)

        try:
            risk_agent = self.get_agent("risk_officer")
            result = await risk_agent.process_request(
                risk_policy, candidate_orders, context
            )

            # Update stats
            orchestration_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self._update_stats(True, orchestration_time)

            return result

        except Exception as e:
            orchestration_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self._update_stats(False, orchestration_time)
            logger.error(f"Assess risk decisions failed: {e}")
            raise

    async def process_complete_workflow(
        self,
        market_context: MarketContext,
        risk_policy: RiskPolicy,
        setups: List[Dict[str, Any]],
        observed_features: ObservedFeatures,
        candidate_orders: List[OrderIntent],
        risk_context: RiskContext
    ) -> Dict[str, Any]:
        """
        Process complete LLM workflow.

        Args:
            market_context: Market context
            risk_policy: Risk policy
            setups: List of setup proposals
            observed_features: Observed market features
            candidate_orders: Candidate orders
            risk_context: Risk context

        Returns:
            Complete workflow results
        """
        start_time = datetime.now(timezone.utc)

        try:
            results = {}

            # Step 1: Validate and rank setups
            if setups:
                results["validations"] = await self.validate_and_rank_setups(
                    market_context, risk_policy, setups
                )

            # Step 2: Propose holistic setups
            if observed_features:
                results["holistic_proposals"] = await self.propose_holistic_setups(
                    market_context, risk_policy, observed_features
                )

            # Step 3: Assess risk decisions
            if candidate_orders:
                results["risk_decisions"] = await self.assess_risk_decisions(
                    risk_policy, candidate_orders, risk_context
                )

            # Update stats
            orchestration_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self._update_stats(True, orchestration_time)

            logger.info("Complete LLM workflow processed successfully")
            return results

        except Exception as e:
            orchestration_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self._update_stats(False, orchestration_time)
            logger.error(f"Complete workflow failed: {e}")
            raise

    def _update_stats(self, success: bool, orchestration_time_ms: float) -> None:
        """Update orchestrator statistics."""
        self.stats["total_orchestrations"] += 1
        self.stats["last_orchestration_time"] = datetime.now(timezone.utc)

        if success:
            self.stats["successful_orchestrations"] += 1
        else:
            self.stats["failed_orchestrations"] += 1

        # Update average orchestration time
        total_orchestrations = self.stats["total_orchestrations"]
        current_avg = self.stats["average_orchestration_time_ms"]
        self.stats["average_orchestration_time_ms"] = (
            (current_avg * (total_orchestrations - 1) + orchestration_time_ms) / total_orchestrations
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        stats = self.stats.copy()

        # Calculate success rate
        if stats["total_orchestrations"] > 0:
            stats["success_rate"] = stats["successful_orchestrations"] / stats["total_orchestrations"]
        else:
            stats["success_rate"] = 0.0

        # Add agent stats
        stats["agents"] = {name: agent.get_stats() for name, agent in self.agents.items()}

        return stats

    def get_all_stats(self) -> Dict[str, Any]:
        """Get all statistics including gateway stats."""
        stats = self.get_stats()
        stats["gateway"] = self.gateway.get_stats()
        stats["validator"] = self.validator.get_validation_stats()
        return stats


# Default configuration
DEFAULT_VALIDATOR_CONFIG = LLMRequestConfig(
    provider="openai",
    model="gpt-4o-mini",
    temperature=0.1,
    max_tokens=1000,
    timeout=30.0,
    retry_attempts=3
)

DEFAULT_PROPOSER_CONFIG = LLMRequestConfig(
    provider="openai",
    model="gpt-4o-mini",
    temperature=0.2,
    max_tokens=1500,
    timeout=30.0,
    retry_attempts=3
)

DEFAULT_RISK_CONFIG = LLMRequestConfig(
    provider="openai",
    model="gpt-4o-mini",
    temperature=0.1,
    max_tokens=800,
    timeout=20.0,
    retry_attempts=3
)


# Global orchestrator instance
llm_orchestrator = LLMOrchestrator()


def get_llm_orchestrator() -> LLMOrchestrator:
    """Get the global LLM orchestrator instance."""
    return llm_orchestrator


def initialize_default_agents() -> None:
    """Initialize default agents with default configurations."""
    orchestrator = get_llm_orchestrator()

    # Register validator agent
    orchestrator.register_agent(
        "validator",
        ValidatorAgent,
        DEFAULT_VALIDATOR_CONFIG
    )

    # Register holistic proposer agent
    orchestrator.register_agent(
        "holistic_proposer",
        HolisticProposerAgent,
        DEFAULT_PROPOSER_CONFIG
    )

    # Register risk officer agent
    orchestrator.register_agent(
        "risk_officer",
        RiskOfficerAgent,
        DEFAULT_RISK_CONFIG
    )

    logger.info("Initialized default LLM agents")