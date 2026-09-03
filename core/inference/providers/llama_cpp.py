"""llama.cpp HTTP Client Provider Adapter.

Connects to a running local llama-server instance via its OpenAI-compatible HTTP API.
Does NOT manage server subprocesses or process lifecycle.
"""

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from core.common.errors import (
    InferenceError,
    ProviderResponseError,
    ProviderUnavailableError,
)
from core.common.types import FinishReason, MessageRole, RuntimeState
from core.inference.provider import BaseProvider
from core.inference.types import (
    InferenceRequest,
    InferenceResponse,
    Message,
    TokenUsage,
)
from core.models.schema import ModelDefinition


class LlamaCppProvider(BaseProvider):
    """Provider adapter communicating with a running llama-server instance over HTTP."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        timeout_seconds: float = 60.0,
    ) -> None:
        """Initialize the llama.cpp HTTP client provider.
        
        Args:
            base_url: Base URL of the llama-server (e.g. 'http://127.0.0.1:8080').
            timeout_seconds: Request timeout in seconds.
        """
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds

    @property
    def provider_name(self) -> str:
        """Unique provider identifier."""
        return "llama_cpp"

    @property
    def base_url(self) -> str:
        """Base URL of the target llama-server."""
        return self._base_url

    @property
    def timeout_seconds(self) -> float:
        """Timeout in seconds for API requests."""
        return self._timeout

    def check_health(self) -> RuntimeState:
        """Probe the llama-server to determine if it is reachable and responsive."""
        # Probe /v1/models
        url = f"{self._base_url}/v1/models"
        req = urllib.request.Request(url, method="GET")
        req.add_header("User-Agent", "LocalAIFoundation/1.0")

        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                if resp.status == 200:
                    return RuntimeState.READY
                return RuntimeState.ERROR
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            return RuntimeState.UNAVAILABLE
        except Exception:
            return RuntimeState.ERROR

    def is_model_loaded(self, model_def: ModelDefinition) -> bool:
        """Check if the llama-server reports this model or its alias as available."""
        url = f"{self._base_url}/v1/models"
        req = urllib.request.Request(url, method="GET")
        req.add_header("User-Agent", "LocalAIFoundation/1.0")

        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                if resp.status != 200:
                    return False
                payload = json.loads(resp.read().decode("utf-8"))
                data = payload.get("data", [])
                server_model_ids = {m.get("id") for m in data if isinstance(m, dict)}

                # Check if canonical ID or any configured alias is reported by the server
                if model_def.id in server_model_ids:
                    return True
                for alias in model_def.aliases:
                    if alias in server_model_ids:
                        return True
                return False
        except Exception:
            return False

    def infer(self, request: InferenceRequest, model_def: ModelDefinition) -> InferenceResponse:
        """Translate normalized request, dispatch to llama-server, and normalize response.
        
        Args:
            request: Normalized inference request.
            model_def: Target model definition from registry.
            
        Returns:
            Normalized InferenceResponse.
        """
        payload = self._build_payload(request, model_def)
        url = f"{self._base_url}/v1/chat/completions"
        json_bytes = json.dumps(payload).encode("utf-8")

        http_req = urllib.request.Request(
            url,
            data=json_bytes,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "LocalAIFoundation/1.0",
            },
            method="POST",
        )

        start_time = time.perf_counter()
        try:
            with urllib.request.urlopen(http_req, timeout=self._timeout) as resp:
                raw_bytes = resp.read()
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                status_code = resp.status
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise InferenceError(
                f"llama-server returned HTTP error {e.code}: {err_body}"
            ) from e
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            raise ProviderUnavailableError(
                f"Failed to connect to llama-server at {self._base_url}: {e}. "
                f"Ensure the server is started via scripts/start_llama_server.sh"
            ) from e
        except Exception as e:
            raise InferenceError(f"Unexpected error communicating with llama-server: {e}") from e

        try:
            resp_dict = json.loads(raw_bytes.decode("utf-8"))
        except Exception as e:
            raise ProviderResponseError(
                f"Failed to parse JSON response from llama-server: {e}"
            ) from e

        return self._normalize_response(
            resp_dict=resp_dict,
            request=request,
            model_def=model_def,
            latency_ms=latency_ms,
        )

    def _resolve_runtime_model_id(self, request: InferenceRequest, model_def: ModelDefinition) -> str:
        """Translate Foundation model identity to runtime-specific model identifier.
        
        Foundation model aliases (e.g. 'default', 'fast') exist at the Foundation
        layer and must not be sent directly to llama-server. By default, the canonical
        model definition ID is used. If metadata specifies a runtime-specific alias
        (e.g. metadata['llama_cpp_alias']), that takes precedence.
        """
        if "llama_cpp_alias" in model_def.metadata:
            return str(model_def.metadata["llama_cpp_alias"])
        return model_def.id

    def _build_payload(self, request: InferenceRequest, model_def: ModelDefinition) -> Dict[str, Any]:
        """Convert normalized InferenceRequest into OpenAI-compatible request payload."""
        messages_list = [
            {"role": msg.role.value, "content": msg.content}
            for msg in request.messages
        ]

        runtime_model = self._resolve_runtime_model_id(request, model_def)
        opts = request.options

        payload: Dict[str, Any] = {
            "model": runtime_model,
            "messages": messages_list,
            "temperature": opts.temperature,
            "top_p": opts.top_p,
            "max_tokens": opts.max_tokens,
            "stream": False,
        }

        if opts.stop_sequences:
            payload["stop"] = opts.stop_sequences

        if opts.seed is not None:
            payload["seed"] = opts.seed

        # Include any extra options configured
        if opts.extra_options:
            payload.update(opts.extra_options)

        # Translate declarative OutputConstraint to provider runtime parameters
        if opts.constraint is not None:
            if opts.constraint.format == "json":
                payload["response_format"] = {"type": "json_object"}
            elif opts.constraint.format == "grammar" and opts.constraint.grammar:
                payload["grammar"] = opts.constraint.grammar

        return payload


    def _normalize_response(
        self,
        resp_dict: Dict[str, Any],
        request: InferenceRequest,
        model_def: ModelDefinition,
        latency_ms: float,
    ) -> InferenceResponse:
        """Convert raw OpenAI-compatible response JSON into normalized InferenceResponse."""
        choices = resp_dict.get("choices")
        if not choices or not isinstance(choices, list):
            raise ProviderResponseError(
                f"Invalid response payload from llama-server: missing or empty 'choices'. "
                f"Payload: {resp_dict}"
            )

        first_choice = choices[0]
        msg_dict = first_choice.get("message", {})
        content = msg_dict.get("content", "")
        role_str = msg_dict.get("role", "assistant")

        try:
            msg_role = MessageRole(role_str)
        except ValueError:
            msg_role = MessageRole.ASSISTANT

        raw_finish = str(first_choice.get("finish_reason", "stop")).lower()
        if raw_finish == "length":
            finish_reason = FinishReason.LENGTH
        elif raw_finish == "stop":
            finish_reason = FinishReason.STOP
        else:
            finish_reason = FinishReason.STOP

        usage_dict = resp_dict.get("usage", {})
        prompt_tokens = int(usage_dict.get("prompt_tokens", 0))
        completion_tokens = int(usage_dict.get("completion_tokens", 0))
        total_tokens = int(usage_dict.get("total_tokens", prompt_tokens + completion_tokens))

        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

        return InferenceResponse(
            request_id=request.request_id or resp_dict.get("id"),
            model_id=model_def.id,
            message=Message(role=msg_role, content=content),
            finish_reason=finish_reason,
            usage=usage,
            latency_ms=latency_ms,
            raw_response=resp_dict,
        )
