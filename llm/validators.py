"""
LLM validators for PSE-LLM trading system.
Strict schema checks and validation utilities.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import BaseModel, ValidationError

from llm.gateway import LLMResponse
from llm.schemas import (
    HolisticInput,
    LLMInput,
    LLMOutput,
    LLMTask,
    ObservedFeatures,
    OrderIntent,
    RiskInput,
)

logger = logging.getLogger(__name__)


class LLMValidationError(Exception):
    """Custom exception for LLM validation errors."""
    pass


class LLMValidator:
    """Validator for LLM inputs and outputs."""

    def __init__(self, strict_mode: bool = True):
        """
        Initialize LLM validator.

        Args:
            strict_mode: Whether to use strict validation
        """
        self.strict_mode = strict_mode
        self.validation_stats = {
            "total_validations": 0,
            "successful_validations": 0,
            "failed_validations": 0,
            "validation_errors": []
        }

    def validate_input(self, data: dict[str, Any], task: LLMTask) -> LLMInput:
        """
        Validate LLM input data.

        Args:
            data: Input data dictionary
            task: LLM task type

        Returns:
            Validated LLM input

        Raises:
            LLMValidationError: If validation fails
        """
        self.validation_stats["total_validations"] += 1

        try:
            # Check required fields
            self._check_required_fields(data, task)

            # Validate data types
            self._validate_data_types(data, task)

            # Create and validate model
            llm_input = LLMInput.model_validate(data)

            # Task-specific validation
            self._validate_task_specific(llm_input, task)

            self.validation_stats["successful_validations"] += 1
            return llm_input

        except (ValidationError, ValueError) as e:
            self.validation_stats["failed_validations"] += 1
            self.validation_stats["validation_errors"].append(str(e))
            raise LLMValidationError(f"Input validation failed: {e}")

    def validate_output(self, data: dict[str, Any], task: LLMTask) -> LLMOutput:
        """
        Validate LLM output data.

        Args:
            data: Output data dictionary
            task: LLM task type

        Returns:
            Validated LLM output

        Raises:
            LLMValidationError: If validation fails
        """
        self.validation_stats["total_validations"] += 1

        try:
            # Check JSON structure
            self._check_json_structure(data)

            # Validate data types
            self._validate_output_data_types(data, task)

            # Create and validate model
            llm_output = LLMOutput.model_validate(data)

            # Task-specific validation
            self._validate_output_task_specific(llm_output, task)

            # Validate numeric constraints
            self._validate_numeric_constraints(llm_output, task)

            self.validation_stats["successful_validations"] += 1
            return llm_output

        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            self.validation_stats["failed_validations"] += 1
            self.validation_stats["validation_errors"].append(str(e))
            raise LLMValidationError(f"Output validation failed: {e}")

    def validate_risk_input(self, data: dict[str, Any]) -> RiskInput:
        """
        Validate risk assessment input data.

        Args:
            data: Risk input data dictionary

        Returns:
            Validated risk input

        Raises:
            LLMValidationError: If validation fails
        """
        self.validation_stats["total_validations"] += 1

        try:
            # Check required fields
            required_fields = ["risk_policy", "candidate_orders", "context"]
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")

            # Create and validate model
            risk_input = RiskInput.model_validate(data)

            # Validate order constraints
            self._validate_order_constraints(risk_input.candidate_orders)

            self.validation_stats["successful_validations"] += 1
            return risk_input

        except (ValidationError, ValueError) as e:
            self.validation_stats["failed_validations"] += 1
            self.validation_stats["validation_errors"].append(str(e))
            raise LLMValidationError(f"Risk input validation failed: {e}")

    def validate_holistic_input(self, data: dict[str, Any]) -> HolisticInput:
        """
        Validate holistic proposal input data.

        Args:
            data: Holistic input data dictionary

        Returns:
            Validated holistic input

        Raises:
            LLMValidationError: If validation fails
        """
        self.validation_stats["total_validations"] += 1

        try:
            # Check required fields
            required_fields = ["session", "market_context", "observed_features", "risk_policy"]
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")

            # Create and validate model
            holistic_input = HolisticInput.model_validate(data)

            # Validate feature constraints
            self._validate_feature_constraints(holistic_input.observed_features)

            self.validation_stats["successful_validations"] += 1
            return holistic_input

        except (ValidationError, ValueError) as e:
            self.validation_stats["failed_validations"] += 1
            self.validation_stats["validation_errors"].append(str(e))
            raise LLMValidationError(f"Holistic input validation failed: {e}")

    def validate_llm_response(self, response: LLMResponse) -> dict[str, Any]:
        """
        Validate and parse LLM response.

        Args:
            response: LLM response

        Returns:
            Parsed response data

        Raises:
            LLMValidationError: If validation fails
        """
        self.validation_stats["total_validations"] += 1

        try:
            # Check response success
            if not response.success:
                raise LLMValidationError(f"LLM request failed: {response.error_message}")

            # Extract JSON from response
            json_data = self._extract_json_from_response(response.content)

            # Validate JSON structure
            self._check_json_structure(json_data)

            self.validation_stats["successful_validations"] += 1
            return json_data

        except (json.JSONDecodeError, ValueError) as e:
            self.validation_stats["failed_validations"] += 1
            self.validation_stats["validation_errors"].append(str(e))
            raise LLMValidationError(f"Response validation failed: {e}")

    def _check_required_fields(self, data: dict[str, Any], task: LLMTask) -> None:
        """Check required fields for input data."""
        required_fields = ["session", "market_context", "setups", "risk_policy", "ask"]

        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        # Validate task-specific requirements
        if task == LLMTask.VALIDATE_AND_RANK and not data.get("setups"):
            raise ValueError("validate_and_rank task requires setups")

    def _validate_data_types(self, data: dict[str, Any], task: LLMTask) -> None:
        """Validate data types for input data."""
        # Validate market context
        market_context = data.get("market_context", {})
        if not isinstance(market_context, dict):
            raise ValueError("market_context must be a dictionary")

        # Validate risk policy
        risk_policy = data.get("risk_policy", {})
        if not isinstance(risk_policy, dict):
            raise ValueError("risk_policy must be a dictionary")

        # Validate setups
        setups = data.get("setups", [])
        if not isinstance(setups, list):
            raise ValueError("setups must be a list")

        # Validate each setup
        for i, setup in enumerate(setups):
            if not isinstance(setup, dict):
                raise ValueError(f"setup {i} must be a dictionary")

    def _validate_task_specific(self, llm_input: LLMInput, task: LLMTask) -> None:
        """Validate task-specific requirements."""
        if task == LLMTask.VALIDATE_AND_RANK:
            if not llm_input.setups:
                raise ValueError("validate_and_rank task requires at least one setup")

            # Validate setup count
            if len(llm_input.setups) > 10:
                raise ValueError("Maximum 10 setups allowed for validation")

        elif task == LLMTask.PROPOSE_HOLISTIC:
            # Validate that setups is empty for holistic proposals
            if llm_input.setups:
                raise ValueError("propose_holistic task should not include setups")

        elif task == LLMTask.RISK_VETO_OR_SCALE:
            # Validate that setups is empty for risk decisions
            if llm_input.setups:
                raise ValueError("risk_veto_or_scale task should not include setups")

    def _check_json_structure(self, data: dict[str, Any]) -> None:
        """Check basic JSON structure."""
        if not isinstance(data, dict):
            raise ValueError("Response must be a JSON object")

        # Check for valid keys
        valid_keys = {"validations", "holistic_proposals", "risk_decisions"}
        for key in data.keys():
            if key not in valid_keys:
                raise ValueError(f"Invalid key in response: {key}")

    def _validate_output_data_types(self, data: dict[str, Any], task: LLMTask) -> None:
        """Validate data types for output data."""
        # Validate validations
        validations = data.get("validations", [])
        if not isinstance(validations, list):
            raise ValueError("validations must be a list")

        for i, validation in enumerate(validations):
            if not isinstance(validation, dict):
                raise ValueError(f"validation {i} must be a dictionary")

        # Validate holistic proposals
        holistic_proposals = data.get("holistic_proposals", [])
        if not isinstance(holistic_proposals, list):
            raise ValueError("holistic_proposals must be a list")

        # Validate risk decisions
        risk_decisions = data.get("risk_decisions", [])
        if not isinstance(risk_decisions, list):
            raise ValueError("risk_decisions must be a list")

        for i, decision in enumerate(risk_decisions):
            if not isinstance(decision, dict):
                raise ValueError(f"risk_decision {i} must be a dictionary")

    def _validate_output_task_specific(self, llm_output: LLMOutput, task: LLMTask) -> None:
        """Validate task-specific output requirements."""
        if task == LLMTask.VALIDATE_AND_RANK:
            if not llm_output.validations:
                raise ValueError("validate_and_rank task requires validations")

            if llm_output.holistic_proposals or llm_output.risk_decisions:
                raise ValueError("validate_and_rank task should only include validations")

        elif task == LLMTask.PROPOSE_HOLISTIC:
            if not llm_output.holistic_proposals:
                raise ValueError("propose_holistic task requires holistic_proposals")

            if llm_output.validations or llm_output.risk_decisions:
                raise ValueError("propose_holistic task should only include holistic_proposals")

        elif task == LLMTask.RISK_VETO_OR_SCALE:
            if not llm_output.risk_decisions:
                raise ValueError("risk_veto_or_scale task requires risk_decisions")

            if llm_output.validations or llm_output.holistic_proposals:
                raise ValueError("risk_veto_or_scale task should only include risk_decisions")

    def _validate_numeric_constraints(self, llm_output: LLMOutput, task: LLMTask) -> None:
        """Validate numeric constraints in output."""
        # Validate validations
        for validation in llm_output.validations:
            if not (0 <= validation.priority <= 1):
                raise ValueError("validation priority must be between 0 and 1")

            if len(validation.reason) > 120:
                raise ValueError("validation reason must be ≤120 characters")

            # Validate scale factor in edits
            if validation.edits and hasattr(validation.edits, 'scale'):
                if validation.edits.scale and not (0 <= validation.edits.scale <= 1):
                    raise ValueError("validation edit scale must be between 0 and 1")

        # Validate risk decisions
        for decision in llm_output.risk_decisions:
            if not (0 <= decision.scale <= 1):
                raise ValueError("risk decision scale must be between 0 and 1")

            if len(decision.reason) > 120:
                raise ValueError("risk decision reason must be ≤120 characters")

    def _validate_order_constraints(self, orders: list[OrderIntent]) -> None:
        """Validate order constraints."""
        for order in orders:
            if order.quantity <= 0:
                raise ValueError("order quantity must be positive")

            if order.entry_price <= 0:
                raise ValueError("order entry price must be positive")

            if order.stop_loss <= 0:
                raise ValueError("order stop loss must be positive")

            if order.estimated_risk < 0:
                raise ValueError("order estimated risk must be non-negative")

    def _validate_feature_constraints(self, features: ObservedFeatures) -> None:
        """Validate feature constraints."""
        if not (0 <= features.mqs <= 10):
            raise ValueError("mqs must be between 0 and 10")

        if not (0 <= features.frs <= 10):
            raise ValueError("frs must be between 0 and 10")

        if features.retracement < 0:
            raise ValueError("retracement must be non-negative")

    def _extract_json_from_response(self, content: str) -> dict[str, Any]:
        """Extract JSON from LLM response content."""
        # Remove markdown code blocks if present
        content = content.strip()

        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]

        if content.endswith("```"):
            content = content[:-3]

        # Try to parse JSON
        try:
            return json.loads(content.strip())
        except json.JSONDecodeError:
            # Try to find JSON object in text
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

            raise LLMValidationError("No valid JSON found in response")

    def get_validation_stats(self) -> dict[str, Any]:
        """Get validation statistics."""
        stats = self.validation_stats.copy()

        # Calculate success rate
        if stats["total_validations"] > 0:
            stats["success_rate"] = stats["successful_validations"] / stats["total_validations"]
        else:
            stats["success_rate"] = 0.0

        return stats

    def reset_stats(self) -> None:
        """Reset validation statistics."""
        self.validation_stats = {
            "total_validations": 0,
            "successful_validations": 0,
            "failed_validations": 0,
            "validation_errors": []
        }


class SchemaValidator:
    """Schema validator for Pydantic models."""

    @staticmethod
    def validate_model(data: dict[str, Any], model_class: type[BaseModel]) -> BaseModel:
        """
        Validate data against Pydantic model.

        Args:
            data: Data to validate
            model_class: Pydantic model class

        Returns:
            Validated model instance

        Raises:
            LLMValidationError: If validation fails
        """
        try:
            return model_class.model_validate(data)
        except ValidationError as e:
            raise LLMValidationError(f"Schema validation failed: {e}")

    @staticmethod
    def validate_list(data: list[dict[str, Any]], model_class: type[BaseModel]) -> list[BaseModel]:
        """
        Validate list of data against Pydantic model.

        Args:
            data: List of data to validate
            model_class: Pydantic model class

        Returns:
            List of validated model instances

        Raises:
            LLMValidationError: If validation fails
        """
        validated_models = []

        for i, item in enumerate(data):
            try:
                validated_models.append(model_class.model_validate(item))
            except ValidationError as e:
                raise LLMValidationError(f"Schema validation failed for item {i}: {e}")

        return validated_models

    @staticmethod
    def is_valid_json(json_str: str) -> bool:
        """
        Check if string is valid JSON.

        Args:
            json_str: JSON string

        Returns:
            True if valid JSON, False otherwise
        """
        try:
            json.loads(json_str)
            return True
        except json.JSONDecodeError:
            return False

    @staticmethod
    def sanitize_json(json_str: str) -> str:
        """
        Sanitize JSON string for parsing.

        Args:
            json_str: JSON string

        Returns:
            Sanitized JSON string
        """
        # Remove common issues
        sanitized = json_str.strip()

        # Remove markdown code blocks
        if sanitized.startswith("```json"):
            sanitized = sanitized[7:]
        elif sanitized.startswith("```"):
            sanitized = sanitized[3:]

        if sanitized.endswith("```"):
            sanitized = sanitized[:-3]

        # Fix common JSON issues
        sanitized = sanitized.replace("'", '"')  # Replace single quotes
        sanitized = re.sub(r',\s*}', '}', sanitized)  # Remove trailing commas
        sanitized = re.sub(r',\s*]', ']', sanitized)  # Remove trailing commas in arrays

        return sanitized.strip()


# Global validator instance
llm_validator = LLMValidator(strict_mode=True)
schema_validator = SchemaValidator()


def get_llm_validator() -> LLMValidator:
    """Get the global LLM validator instance."""
    return llm_validator


def get_schema_validator() -> SchemaValidator:
    """Get the global schema validator instance."""
    return schema_validator
