"""PydanticAI Model adapter bridging PydanticAI to FoundationInferenceConnector.

Provides FoundationPydanticAIModel which subclasses PydanticAI's Model interface,
translating requests and messages into normalized Foundation contracts while
enforcing ModelSelectionPolicy tiering and preserving hard runtime identity checks.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
import base64
import uuid

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage

from connectors.inference import InferenceConnector
from core.common.types import MessageRole
from core.inference.types import (
    GenerationOptions,
    InferenceRequest,
    InferenceResponse,
    MediaAttachment,
    Message,
)
from orchestration.routing.model_selector import ModelSelectionPolicy
from orchestration.routing.types import ModelTier

logger = logging.getLogger(__name__)


class FoundationPydanticAIModel(Model):
    """Local PydanticAI Model adapter backed by FoundationInferenceConnector.

    Decoupled from llama.cpp, CUDA, and concrete model IDs. Resolves abstract
    ModelTier through ModelSelectionPolicy and delegates execution to InferenceConnector.
    """

    def __init__(
        self,
        connector: InferenceConnector,
        model_policy: ModelSelectionPolicy,
        default_tier: ModelTier = ModelTier.REASONING,
    ) -> None:
        super().__init__()
        self._connector = connector
        self._model_policy = model_policy
        self._current_tier = default_tier

    @property
    def model_name(self) -> str:
        """Resolved concrete model ID for the active tier."""
        return self._model_policy.resolve_model_id(self._current_tier)

    @property
    def system(self) -> str | None:
        """Provider identifier for PydanticAI telemetry."""
        return "foundation"

    @property
    def current_tier(self) -> ModelTier:
        """Currently selected ModelTier."""
        return self._current_tier

    def set_tier(self, tier: ModelTier) -> None:
        """Set the active ModelTier."""
        self._current_tier = tier

    def _translate_messages(self, messages: List[ModelMessage]) -> List[Message]:
        """Translate PydanticAI ModelMessage objects into Foundation Messages."""
        foundation_messages: List[Message] = []

        for msg in messages:
            if isinstance(msg, ModelRequest):
                attachments: List[MediaAttachment] = []
                text_parts: List[str] = []

                for part in msg.parts:
                    if isinstance(part, SystemPromptPart):
                        foundation_messages.append(
                            Message(role=MessageRole.SYSTEM, content=str(part.content))
                        )
                    elif isinstance(part, ToolReturnPart):
                        content_val = part.content
                        if not isinstance(content_val, str):
                            try:
                                content_str = json.dumps(content_val)
                            except Exception:
                                content_str = str(content_val)
                        else:
                            content_str = content_val
                        foundation_messages.append(
                            Message(
                                role=MessageRole.TOOL,
                                content=content_str,
                                name=part.tool_name,
                            )
                        )
                    elif isinstance(part, UserPromptPart):
                        if isinstance(part.content, str):
                            text_parts.append(part.content)
                        elif isinstance(part.content, (list, tuple)):
                            for item in part.content:
                                if isinstance(item, str):
                                    text_parts.append(item)
                                elif hasattr(item, "data") and hasattr(item, "media_type"):
                                    attachments.append(
                                        MediaAttachment.from_bytes(
                                            data=item.data,
                                            mime_type=item.media_type,
                                        )
                                    )
                                elif hasattr(item, "url"):
                                    url_val = item.url
                                    if url_val.startswith("data:"):
                                        header, _, b64_data = url_val.partition(",")
                                        mime = header.split(";")[0].replace("data:", "") or "image/png"
                                        raw = base64.b64decode(b64_data)
                                        attachments.append(MediaAttachment.from_bytes(raw, mime_type=mime))
                                    else:
                                        local_path = url_val.replace("file://", "")
                                        attachments.append(MediaAttachment.from_file(local_path))
                                else:
                                    text_parts.append(str(item))
                        else:
                            text_parts.append(str(part.content))
                    elif hasattr(part, "data") and hasattr(part, "media_type"):
                        attachments.append(
                            MediaAttachment.from_bytes(
                                data=part.data,
                                mime_type=part.media_type,
                            )
                        )
                    elif hasattr(part, "url"):
                        url_val = part.url
                        if url_val.startswith("data:"):
                            header, _, b64_data = url_val.partition(",")
                            mime = header.split(";")[0].replace("data:", "") or "image/png"
                            raw = base64.b64decode(b64_data)
                            attachments.append(MediaAttachment.from_bytes(raw, mime_type=mime))
                        else:
                            local_path = url_val.replace("file://", "")
                            attachments.append(MediaAttachment.from_file(local_path))
                    else:
                        content = getattr(part, "content", str(part))
                        text_parts.append(str(content))

                if text_parts or attachments:
                    user_content = "\n".join(text_parts) if text_parts else ""
                    foundation_messages.append(
                        Message(
                            role=MessageRole.USER,
                            content=user_content,
                            attachments=tuple(attachments),
                        )
                    )

            elif isinstance(msg, ModelResponse):
                for part in msg.parts:
                    if isinstance(part, TextPart):
                        foundation_messages.append(
                            Message(role=MessageRole.ASSISTANT, content=str(part.content))
                        )
                    elif isinstance(part, ToolCallPart):
                        args = part.args
                        if not isinstance(args, str):
                            try:
                                args_str = json.dumps(args)
                            except Exception:
                                args_str = str(args)
                        else:
                            args_str = args
                        call_payload = json.dumps({
                            "tool": part.tool_name,
                            "arguments": args,
                            "tool_call_id": part.tool_call_id,
                        })
                        foundation_messages.append(
                            Message(role=MessageRole.ASSISTANT, content=call_payload)
                        )

        return foundation_messages

    def _parse_tool_calls_or_text(
        self,
        resp: InferenceResponse,
        tool_defs: Dict[str, Any],
    ) -> List[Any]:
        """Detect and parse tool call proposals or plain text from model output."""
        # 1. Check raw provider response for structured OpenAI-format tool_calls
        if resp.raw_response and isinstance(resp.raw_response, dict):
            choices = resp.raw_response.get("choices", [])
            if choices and isinstance(choices, list) and len(choices) > 0:
                msg_dict = choices[0].get("message", {})
                raw_tool_calls = msg_dict.get("tool_calls", [])
                if raw_tool_calls and isinstance(raw_tool_calls, list):
                    parts = []
                    for tc in raw_tool_calls:
                        fn = tc.get("function", {})
                        name = fn.get("name", "")
                        raw_args = fn.get("arguments", "{}")
                        try:
                            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                        except Exception:
                            args = {"raw": raw_args}
                        call_id = tc.get("id") or f"call-{uuid.uuid4().hex[:8]}"
                        parts.append(ToolCallPart(tool_name=name, args=args, tool_call_id=call_id))
                    if parts:
                        return parts

        # 2. Check text content for JSON tool call syntax if tool_defs are available
        text = resp.text.strip()
        if tool_defs and (text.startswith("{") and text.endswith("}")):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    # Check shapes like {"tool": "name", "arguments": {...}}
                    tool_name = parsed.get("tool") or parsed.get("name") or parsed.get("tool_name")
                    args = parsed.get("arguments") or parsed.get("args") or parsed.get("parameters")
                    if tool_name:
                        call_id = parsed.get("tool_call_id") or f"call-{uuid.uuid4().hex[:8]}"
                        return [
                            ToolCallPart(
                                tool_name=str(tool_name),
                                args=args if isinstance(args, dict) else {},
                                tool_call_id=call_id,
                            )
                        ]
            except Exception:
                pass

        # 3. Default to TextPart
        return [TextPart(content=resp.text)]

    async def request(
        self,
        messages: List[ModelMessage],
        model_settings: Optional[ModelSettings],
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Translate request to Foundation contracts, invoke InferenceConnector, and return ModelResponse."""
        concrete_model_id = self.model_name
        foundation_messages = self._translate_messages(messages)

        # Map generation settings
        temp = 0.7
        max_tokens = 1024
        if model_settings is not None:
            if model_settings.temperature is not None:
                temp = float(model_settings.temperature)
            if model_settings.max_tokens is not None:
                max_tokens = int(model_settings.max_tokens)

        extra_opts: Dict[str, Any] = {}
        if model_request_parameters.tool_defs:
            tools_payload = []
            for t in model_request_parameters.tool_defs.values():
                tools_payload.append({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": t.parameters_json_schema,
                    },
                })
            extra_opts["tools"] = tools_payload

        options = GenerationOptions(
            temperature=temp,
            max_tokens=max_tokens,
            extra_options=extra_opts or None,
        )

        request = InferenceRequest(
            model_id=concrete_model_id,
            messages=foundation_messages,
            options=options,
            request_id=f"pydantic-ai-{uuid.uuid4().hex[:8]}",
        )

        # Dispatch inference to connector in a thread pool to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        resp: InferenceResponse = await loop.run_in_executor(
            None,
            self._connector.infer,
            request,
        )

        tool_defs = model_request_parameters.tool_defs or {}
        parts = self._parse_tool_calls_or_text(resp, tool_defs)

        usage = RequestUsage(
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
        )

        return ModelResponse(
            parts=parts,
            usage=usage,
            model_name=concrete_model_id,
        )
