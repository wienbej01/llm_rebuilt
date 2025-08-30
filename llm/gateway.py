"""
LLM Gateway for PSE-LLM trading system.
Provider-agnostic async HTTP client for LLM communication.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class LLMModel(str, Enum):
    """Supported LLM models."""
    # OpenAI models
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_3_5_TURBO = "gpt-3.5-turbo"

    # Anthropic models
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"

    # Gemini models
    GEMINI_1_5_PRO = "gemini-1.5-pro"
    GEMINI_1_5_FLASH = "gemini-1.5-flash"


class LLMRequest(BaseModel):
    """LLM request model."""
    provider: LLMProvider
    model: LLMModel
    messages: list[dict[str, str]]
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=1000, ge=1)
    timeout: float = Field(default=30.0, ge=1.0)
    retry_attempts: int = Field(default=3, ge=0)

    model_config = ConfigDict(frozen=True)


class LLMResponse(BaseModel):
    """LLM response model."""
    provider: LLMProvider
    model: LLMModel
    content: str
    usage: dict[str, int]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    response_time_ms: float
    success: bool
    error_message: str | None = None

    model_config = ConfigDict(frozen=True)


class LLMGateway:
    """Provider-agnostic LLM gateway with async HTTP client."""

    def __init__(
        self,
        default_provider: LLMProvider = LLMProvider.OPENAI,
        default_model: LLMModel = LLMModel.GPT_4O_MINI,
        timeout: float = 30.0,
        retry_attempts: int = 3,
        max_concurrent_requests: int = 10
    ):
        """
        Initialize LLM gateway.

        Args:
            default_provider: Default LLM provider
            default_model: Default LLM model
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts
            max_concurrent_requests: Maximum concurrent requests
        """
        self.default_provider = default_provider
        self.default_model = default_model
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.max_concurrent_requests = max_concurrent_requests

        # HTTP client
        self.client = httpx.AsyncClient(timeout=timeout)

        # Rate limiting semaphore
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

        # API keys (should be loaded from environment)
        self.api_keys = {
            LLMProvider.OPENAI: None,
            LLMProvider.ANTHROPIC: None,
            LLMProvider.GEMINI: None
        }

        # Request statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time_ms": 0.0,
            "last_request_time": None
        }

        logger.info(f"LLM Gateway initialized with {default_provider} / {default_model}")

    def set_api_key(self, provider: LLMProvider, api_key: str) -> None:
        """
        Set API key for a provider.

        Args:
            provider: LLM provider
            api_key: API key
        """
        self.api_keys[provider] = api_key
        logger.info(f"Set API key for {provider}")

    async def send_request(
        self,
        request: LLMRequest,
        api_key: str | None = None
    ) -> LLMResponse:
        """
        Send request to LLM provider.

        Args:
            request: LLM request
            api_key: Optional API key override

        Returns:
            LLM response
        """
        async with self.semaphore:
            start_time = datetime.now(UTC)

            try:
                # Get API key
                key = api_key or self.api_keys.get(request.provider)
                if not key:
                    raise ValueError(f"No API key provided for {request.provider}")

                # Send request based on provider
                if request.provider == LLMProvider.OPENAI:
                    response = await self._send_openai_request(request, key)
                elif request.provider == LLMProvider.ANTHROPIC:
                    response = await self._send_anthropic_request(request, key)
                elif request.provider == LLMProvider.GEMINI:
                    response = await self._send_gemini_request(request, key)
                else:
                    raise ValueError(f"Unsupported provider: {request.provider}")

                # Update statistics
                self._update_stats(response.success, response.response_time_ms)

                return response

            except Exception as e:
                response_time = (datetime.now(UTC) - start_time).total_seconds() * 1000
                error_response = LLMResponse(
                    provider=request.provider,
                    model=request.model,
                    content="",
                    usage={},
                    response_time_ms=response_time,
                    success=False,
                    error_message=str(e)
                )

                self._update_stats(False, response_time)
                logger.error(f"LLM request failed: {e}")

                return error_response

    async def _send_openai_request(self, request: LLMRequest, api_key: str) -> LLMResponse:
        """Send request to OpenAI."""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": request.model.value,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens
        }

        response = await self._retry_request(
            "POST",
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            max_attempts=request.retry_attempts
        )

        if response.status_code == 200:
            data = response.json()
            return LLMResponse(
                provider=LLMProvider.OPENAI,
                model=request.model,
                content=data["choices"][0]["message"]["content"],
                usage=data.get("usage", {}),
                response_time_ms=response.elapsed.total_seconds() * 1000,
                success=True
            )
        else:
            raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")

    async def _send_anthropic_request(self, request: LLMRequest, api_key: str) -> LLMResponse:
        """Send request to Anthropic."""
        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }

        # Convert messages to Anthropic format
        system_message = ""
        user_messages = []

        for msg in request.messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                user_messages.append(msg)

        payload = {
            "model": request.model.value,
            "max_tokens": request.max_tokens or 1000,
            "temperature": request.temperature,
            "system": system_message,
            "messages": user_messages
        }

        response = await self._retry_request(
            "POST",
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            max_attempts=request.retry_attempts
        )

        if response.status_code == 200:
            data = response.json()
            return LLMResponse(
                provider=LLMProvider.ANTHROPIC,
                model=request.model,
                content=data["content"][0]["text"],
                usage=data.get("usage", {}),
                response_time_ms=response.elapsed.total_seconds() * 1000,
                success=True
            )
        else:
            raise Exception(f"Anthropic API error: {response.status_code} - {response.text}")

    async def _send_gemini_request(self, request: LLMRequest, api_key: str) -> LLMResponse:
        """Send request to Gemini."""
        headers = {
            "Content-Type": "application/json"
        }

        # Convert messages to Gemini format
        system_message = ""
        user_messages = []

        for msg in request.messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                user_messages.append(msg)

        # Gemini format
        contents = []
        for msg in user_messages:
            if msg["role"] == "user":
                contents.append({"role": "user", "parts": [{"text": msg["content"]}]})
            elif msg["role"] == "assistant":
                contents.append({"role": "model", "parts": [{"text": msg["content"]}]})

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": request.temperature,
                "maxOutputTokens": request.max_tokens or 1000
            }
        }

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{request.model.value}:generateContent?key={api_key}"

        response = await self._retry_request(
            "POST",
            url,
            headers=headers,
            json=payload,
            max_attempts=request.retry_attempts
        )

        if response.status_code == 200:
            data = response.json()
            content = ""
            if "candidates" in data and data["candidates"]:
                content = data["candidates"][0]["content"]["parts"][0]["text"]

            return LLMResponse(
                provider=LLMProvider.GEMINI,
                model=request.model,
                content=content,
                usage=data.get("usageMetadata", {}),
                response_time_ms=response.elapsed.total_seconds() * 1000,
                success=True
            )
        else:
            raise Exception(f"Gemini API error: {response.status_code} - {response.text}")

    async def _retry_request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        json: dict[str, Any],
        max_attempts: int
    ) -> httpx.Response:
        """Retry HTTP request with exponential backoff."""
        for attempt in range(max_attempts + 1):
            try:
                response = await self.client.request(method, url, headers=headers, json=json)
                return response
            except Exception as e:
                if attempt == max_attempts:
                    raise

                # Exponential backoff
                wait_time = 2 ** attempt
                logger.warning(f"Request failed (attempt {attempt + 1}/{max_attempts + 1}), retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)

    def _update_stats(self, success: bool, response_time_ms: float) -> None:
        """Update request statistics."""
        self.stats["total_requests"] += 1
        self.stats["last_request_time"] = datetime.now(UTC)

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

    async def send_json_request(
        self,
        provider: LLMProvider,
        model: LLMModel,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        timeout: float | None = None,
        retry_attempts: int | None = None
    ) -> LLMResponse:
        """
        Send JSON-formatted request to LLM.

        Args:
            provider: LLM provider
            model: LLM model
            system_prompt: System prompt
            user_prompt: User prompt
            temperature: Temperature
            max_tokens: Max tokens
            timeout: Timeout
            retry_attempts: Retry attempts

        Returns:
            LLM response
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        request = LLMRequest(
            provider=provider,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout or self.timeout,
            retry_attempts=retry_attempts or self.retry_attempts
        )

        return await self.send_request(request)

    async def validate_json_response(
        self,
        response: LLMResponse,
        schema: BaseModel | None = None
    ) -> dict[str, Any]:
        """
        Validate and parse JSON response from LLM.

        Args:
            response: LLM response
            schema: Optional Pydantic schema for validation

        Returns:
            Parsed JSON data

        Raises:
            ValueError: If response is not valid JSON
        """
        if not response.success:
            raise ValueError(f"LLM request failed: {response.error_message}")

        try:
            # Extract JSON from response (handle markdown code blocks)
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]

            data = json.loads(content)

            if schema:
                return schema.model_validate(data)

            return data

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {e}")
        except Exception as e:
            raise ValueError(f"Schema validation failed: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get gateway statistics."""
        return self.stats.copy()

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()
        logger.info("LLM Gateway closed")

    async def __aenter__(self) -> LLMGateway:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()


# Global gateway instance
llm_gateway = LLMGateway()


def get_llm_gateway() -> LLMGateway:
    """Get the global LLM gateway instance."""
    return llm_gateway
