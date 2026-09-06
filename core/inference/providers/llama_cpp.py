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
    ConfigurationError,
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

RESERVED_PAYLOAD_KEYS = frozenset({
    "model",
    "messages",
    "stream",
    "temperature",
    "top_p",
    "max_tokens",
    "stop",
    "seed",
    "response_format",
    "grammar",
})


class LlamaCppProvider(BaseProvider):
    """Provider adapter communicating with a running llama-server instance over HTTP."""


    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        timeout_seconds: float = 60.0,
        max_response_bytes: int = 10 * 1024 * 1024,
    ) -> None:
        """Initialize the llama.cpp HTTP client provider.
        
        Args:
            base_url: Base URL of the llama-server (e.g. 'http://127.0.0.1:8080').
            timeout_seconds: Request timeout in seconds.
            max_response_bytes: Maximum allowed response size in bytes (defaults to 10 MiB).
        """
        if not isinstance(base_url, str) or not (base_url.startswith("http://") or base_url.startswith("https://")):
            raise ConfigurationError(
                f"Invalid base_url '{base_url}'. LlamaCppProvider requires an 'http://' or 'https://' URL."
            )

        if isinstance(max_response_bytes, bool) or not isinstance(max_response_bytes, int) or max_response_bytes <= 0:
            raise ConfigurationError(
                f"max_response_bytes must be a positive integer, got {max_response_bytes!r}"
            )

        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._max_response_bytes = max_response_bytes

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

    @property
    def max_response_bytes(self) -> int:
        """Maximum response body size in bytes."""
        return self._max_response_bytes


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
                if "llama_cpp_alias" in model_def.metadata and model_def.metadata["llama_cpp_alias"] in server_model_ids:
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
                raw_bytes = resp.read(self._max_response_bytes + 1)
                if len(raw_bytes) > self._max_response_bytes:
                    raise ProviderResponseError(
                        f"Response from llama-server exceeded maximum allowed size of {self._max_response_bytes} bytes."
                    )
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                status_code = resp.status
        except ProviderResponseError:
            raise
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
        """Construct the llama-server /v1/chat/completions payload from a normalized request."""
        messages_list: List[Dict[str, Any]] = []
        for msg in request.messages:
            if not getattr(msg, "attachments", None):
                messages_list.append({"role": msg.role.value, "content": msg.content})
            else:
                content_blocks: List[Dict[str, Any]] = []
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})
                for att in msg.attachments:
                    try:
                        data_uri = att.to_base64_data_uri()
                    except Exception as e:
                        raise InferenceError(
                            f"Failed to load and serialize media attachment '{att.name or att.source_path}': {e}"
                        ) from e
                    content_blocks.append({
                        "type": "image_url",
                        "image_url": {"url": data_uri},
                    })
                messages_list.append({"role": msg.role.value, "content": content_blocks})

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

        # Include any extra options configured (protecting reserved normalized keys)
        if opts.extra_options:
            collisions = set(opts.extra_options.keys()) & RESERVED_PAYLOAD_KEYS
            if collisions:
                raise InferenceError(
                    f"extra_options cannot override normalized parameter(s): {sorted(collisions)}. "
                    f"Use the corresponding fields on GenerationOptions instead."
                )
            payload.update(opts.extra_options)

        # Translate declarative OutputConstraint to provider runtime parameters
        if opts.constraint is not None:
            if opts.constraint.format == "json":
                payload["response_format"] = {"type": "json_object"}
            elif opts.constraint.format == "grammar":
                if not opts.constraint.grammar or not opts.constraint.grammar.strip():
                    raise InferenceError(
                        "OutputConstraint with format='grammar' requires non-empty grammar text."
                    )
                payload["grammar"] = opts.constraint.grammar
            else:
                raise InferenceError(
                    f"LlamaCppProvider does not support OutputConstraint format '{opts.constraint.format}'. "
                    f"Supported formats are: 'json', 'grammar'."
                )

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

        usage_dict = resp_dict.get("usage")
        if not isinstance(usage_dict, dict):
            usage_dict = {}

        def _parse_token_count(data: Dict[str, Any], key: str) -> int:
            if key not in data or data[key] is None:
                return 0
            val = data[key]
            if isinstance(val, bool):
                raise ProviderResponseError(f"Invalid boolean value for token usage '{key}': {val!r}")
            try:
                parsed = int(val)
                if parsed < 0:
                    raise ProviderResponseError(f"Negative token usage value for '{key}': {parsed}")
                return parsed
            except (ValueError, TypeError) as exc:
                raise ProviderResponseError(
                    f"Malformed non-numeric token usage value for '{key}': {val!r}"
                ) from exc

        prompt_tokens = _parse_token_count(usage_dict, "prompt_tokens")
        completion_tokens = _parse_token_count(usage_dict, "completion_tokens")
        if "total_tokens" in usage_dict and usage_dict["total_tokens"] is not None:
            total_tokens = _parse_token_count(usage_dict, "total_tokens")
        else:
            total_tokens = prompt_tokens + completion_tokens

        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

        # Validate that the response returned by llama-server actually matches the requested model
        returned_model = resp_dict.get("model")
        if returned_model:
            expected_runtime_id = self._resolve_runtime_model_id(request, model_def)
            if not (model_def.matches_identifier(str(returned_model)) or str(returned_model) == expected_runtime_id):
                raise ProviderResponseError(
                    f"LlamaCppProvider executed wrong model: requested '{model_def.id}' "
                    f"(expected runtime model '{expected_runtime_id}'), "
                    f"but llama-server reported execution of model '{returned_model}'."
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
