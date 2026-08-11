"""
llm‑lmstudio — enhanced plugin
Requires llm >= 0.32 (structured messages API) and LM Studio >= 0.4.0.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from collections.abc import AsyncGenerator, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, ClassVar, cast
from urllib.parse import urlparse

import httpx
import llm
import requests
from llm.parts import (
    AttachmentPart,
    ReasoningPart,
    StreamEvent,
    TextPart,
    ToolCallPart,
    ToolResultPart,
)
from pydantic import Field

# --------------------------------------------------------------------------- #
#  Configuration                                                              #
# --------------------------------------------------------------------------- #
raw = (
    os.getenv("LMSTUDIO_API_BASE")  # singular, supports comma-separated list
    or "http://localhost:1234"
)  # hard default
SERVER_LIST = [u.strip().rstrip("/") for u in raw.split(",") if u.strip()]
TIMEOUT = float(os.getenv("LMSTUDIO_TIMEOUT", "90"))

# --------------------------------------------------------------------------- #
#  Internal helpers                                                           #
# --------------------------------------------------------------------------- #
_cache: dict[str, tuple[list[dict[str, Any]], str]] = {}
_errors: dict[str, Exception] = {}


def _debug(message: str) -> None:
    if os.getenv("LLM_LMSTUDIO_DEBUG") == "1":
        print(message, file=sys.stderr)


def _fetch_models(base: str) -> tuple[list[dict[str, Any]], str]:
    """Return cached metadata and API path prefix for one LM Studio server."""
    if base in _cache:
        return _cache[base]
    try:
        # Prefer the richer metadata endpoint
        api_path = "/api/v0"
        _debug(f"LMSTUDIO DEBUG: Fetching models from {base}{api_path}/models")
        r = requests.get(f"{base}{api_path}/models", timeout=TIMEOUT)
        if r.status_code == 404:  # Older LM Studio → fall back
            api_path = "/v1"
            _debug(
                f"LMSTUDIO DEBUG: {base}/api/v0/models not found, falling back to {base}{api_path}/models"
            )
            r = requests.get(f"{base}{api_path}/models", timeout=TIMEOUT)
            r.raise_for_status()
            data = r.json().get("data", [])
            _debug(
                f"LMSTUDIO DEBUG: Received data from /v1 endpoint for {base}: {data}"
            )
            # v1 has no 'type'; assume plain LLM or infer from ID
            meta = []
            for m_data in data:
                m_id = m_data["id"] if isinstance(m_data, dict) else m_data
                m_type = "embeddings" if "embed" in m_id.lower() else "llm"
                # V1 doesn't reliably tell us VLM status
                current_model_meta = {"id": m_id, "type": m_type, "vision": False}
                _debug(
                    f"LMSTUDIO DEBUG: Processed /v1 model data for {base}: ID={m_id}, Inferred Type={m_type}, Vision=False"
                )
                meta.append(current_model_meta)
        else:
            r.raise_for_status()
            meta = r.json().get("data", [])
            _debug(f"LMSTUDIO DEBUG: Received full metadata from /api/v0 for {base}:")
            for m_debug in meta:
                _debug(
                    f"  LMSTUDIO DEBUG: Model ID: {m_debug.get('id')}, Type: {m_debug.get('type')}, Original Vision: {m_debug.get('vision')}, Path: {m_debug.get('path')}, Publisher: {m_debug.get('publisher')}, Architecture: {m_debug.get('architecture')}, Quantization: {m_debug.get('quantization')}"
                )

            # Add 'vision' flag for /api/v0 models for clarity
            for m in meta:
                # Explicitly check if type is 'vlm'
                is_vlm_type = m.get("type") == "vlm"
                # Check if 'vision' key exists and is true
                has_vision_flag = m.get("vision") is True  # Explicitly check for True

                m["vision"] = (
                    is_vlm_type or has_vision_flag
                )  # Set our 'vision' flag based on these
                _debug(
                    f"  LMSTUDIO DEBUG: For model {m.get('id')}: Original type='{m.get('type')}', original vision_key_present_and_true='{has_vision_flag}', calculated plugin vision_support='{m['vision']}'"
                )

        _cache[base] = (meta, api_path)
        return meta, api_path
    except Exception as e:
        _errors[base] = e
        return [], ""  # Return empty list and empty path on error


def _host_tag(base: str) -> str:
    """Turn 'http://192.168.1.40:1234' into '192_168_1_40_1234'."""
    return urlparse(base).netloc.replace(":", "_").replace(".", "_")


# --------------------------------------------------------------------------- #
#  Registration hooks                                                         #
# --------------------------------------------------------------------------- #
@llm.hookimpl
def register_models(register):
    single_server = len(SERVER_LIST) == 1
    for base in SERVER_LIST:
        models, api_path = _fetch_models(base)
        if not models and not api_path:  # Skip if fetch failed completely
            continue
        for m in models:
            if m.get("type") == "embeddings":
                continue  # handled in embedding hook

            raw_id = m["id"]
            if single_server:
                model_id = f"lmstudio/{raw_id}"
            else:
                model_id = f"lmstudio@{_host_tag(base)}/{raw_id}"

            # Check if model is loaded (only reliable via /api/v0)
            is_loaded = m.get("state") == "loaded"
            # display_model_id = f"{model_id} 🟢" if is_loaded else model_id

            # Use the 'vision' flag that was calculated and refined by _fetch_models
            supports_images_flag = m.get("vision", False)

            # Modify the actual model_id that will be registered
            # final_model_id_for_registration = model_id # This was the old approach
            # The actual model_id should remain clean for lookup.

            display_suffix_parts = []
            if is_loaded:
                display_suffix_parts.append("●")  # Black Circle for loaded
            if supports_images_flag:
                display_suffix_parts.append("👁")  # Eye for vision (U+1F441)
            # Assuming schema support for these models, as it's a general capability of the endpoint
            display_suffix_parts.append(
                "⚒"
            )  # Hammer and Pick for schema/tools (U+2692)

            display_suffix = ""
            if display_suffix_parts:
                display_suffix = " " + " ".join(
                    display_suffix_parts
                )  # Join with spaces, add leading space

            _debug(
                f"LMSTUDIO DEBUG [register_models]: Base model_id: '{model_id}', Calculated display_suffix: '{display_suffix}'"
            )
            _debug(
                f"LMSTUDIO DEBUG [register_models]: For {raw_id}, passing model_id='{model_id}', supports_images={supports_images_flag} to constructor."
            )

            current_metadata = {
                "publisher": m.get("publisher"),
                "arch": m.get(
                    "arch"
                ),  # Note: LM Studio API docs say 'architecture' but example shows 'arch'
                "quantization": m.get("quantization"),
                "max_context_length": m.get("max_context_length"),
                "state": m.get(
                    "state", "unknown"
                ),  # Default to unknown if not available
                "api_path": api_path,  # Store the API path used
                "base_url": base,  # Store the base URL
                "vision": supports_images_flag,  # Ensure our calculated flag is in metadata
                "raw_lmstudio_type": m.get(
                    "type"
                ),  # Store original type for debugging/inspection
            }
            # Filter out None values for cleaner inspection, but keep 'vision' as it's boolean
            current_metadata = {
                k: v
                for k, v in current_metadata.items()
                if v is not None or k == "vision"
            }

            register(
                LMStudioModel(
                    model_id,
                    base,
                    raw_id,
                    api_path,
                    supports_images=supports_images_flag,
                    metadata=current_metadata,
                    display_suffix=display_suffix,
                ),
                LMStudioAsyncModel(
                    model_id,
                    base,
                    raw_id,
                    api_path,
                    supports_images=supports_images_flag,
                    metadata=current_metadata,
                    display_suffix=display_suffix,
                ),
            )
    if _errors:
        _debug(
            "Warning: Some LM Studio servers were unreachable:\n  "
            + "\n  ".join(f"{k}: {v}" for k, v in _errors.items())
        )


@llm.hookimpl
def register_embedding_models(register):
    single_server = len(SERVER_LIST) == 1
    for base in SERVER_LIST:
        models, api_path = _fetch_models(base)
        if not models and not api_path:  # Skip if fetch failed completely
            continue
        for m in models:
            if m.get("type") == "embeddings":
                raw_id = m["id"]
                if single_server:
                    model_id = raw_id
                else:
                    model_id = f"lmstudio@{_host_tag(base)}/{raw_id}"
                register(LMStudioEmbeddingModel(model_id, base, raw_id, api_path))


# --------------------------------------------------------------------------- #
#  Model classes                                                              #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ChatRequest:
    url: str
    payload: dict[str, Any]
    stream: bool
    timeout: float


@dataclass
class StreamState:
    chunks: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] | None = None


class LMStudioBaseModel:
    """Base class for common LMStudio model attributes."""

    can_stream: bool = True
    supports_tools: bool = True
    attachment_types: ClassVar[set[str]] = {
        "image/png",
        "image/jpeg",
        "image/gif",
        "image/webp",
    }

    def __init__(
        self,
        model_id: str,
        base_url: str,
        raw_id: str,
        api_path_prefix: str,
        *,
        supports_images: bool = False,
        metadata: dict | None = None,
        display_suffix: str = "",
    ):
        self.model_id = model_id
        self.raw_id = raw_id
        self.base = base_url
        self.api_path_prefix = api_path_prefix
        self.supports_images = supports_images
        self.metadata = metadata or {}
        self.supports_schema = True
        self.display_suffix = display_suffix

    class Options(llm.Options):
        temperature: float | None = Field(None, description="Sampling temperature")
        top_p: float | None = Field(None, description="Nucleus sampling")
        max_tokens: int | None = Field(None, description="Maximum tokens")
        stop: list[str] | None = Field(None, description="Stop sequences")

    def _prepare_chat_request(
        self,
        prompt: llm.Prompt,
        stream: bool,
        conversation=None,
    ) -> ChatRequest:
        has_schema = bool(getattr(prompt, "schema", None))
        has_tools = bool(getattr(prompt, "tools", None))

        url = (
            f"{self.base}/v1/chat/completions"
            if has_schema
            else f"{self.base}{self.api_path_prefix}/chat/completions"
        )
        payload: dict[str, Any] = {
            "model": self.raw_id,
            "messages": self._build_messages(prompt, conversation),
        }

        if has_tools:
            payload["tools"] = self._encode_tools(prompt.tools)

        if prompt.options:
            payload.update(prompt.options.model_dump(exclude_none=True))

        if has_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "llm_generated_schema",
                    "schema": prompt.schema,
                },
            }
            stream = False

        payload["stream"] = stream
        if stream:
            payload["stream_options"] = {"include_usage": True}

        timeout = TIMEOUT
        if has_schema:
            timeout = max(TIMEOUT, 30.0)
        elif has_tools:
            timeout = max(TIMEOUT, 15.0)

        return ChatRequest(
            url=url,
            payload=payload,
            stream=stream,
            timeout=timeout,
        )

    def __str__(self):
        """Return the model ID with its display suffix for listings."""
        return f"{self.model_id}{self.display_suffix}"

    def inspect(self):
        """Return model metadata for the 'llm inspect' command."""
        return self.metadata

    # --------------------------------------------------------------------- #
    #  Check/Load Helpers                                                   #
    # --------------------------------------------------------------------- #
    def _is_model_loaded(self) -> bool:
        """Check if the current model is loaded."""
        if self.api_path_prefix == "/api/v0":
            try:
                # Use the specific model endpoint if available (/api/v0)
                url = f"{self.base}{self.api_path_prefix}/models/{self.raw_id}"
                r = requests.get(url, timeout=TIMEOUT)
                if r.status_code == 200:
                    return r.json().get("state") == "loaded"
                elif (
                    r.status_code == 404
                ):  # Model exists but endpoint doesn't? Unlikely but handle
                    pass  # Fallback to checking /v1/models list
                else:
                    r.raise_for_status()  # Raise other errors
            except requests.RequestException:
                pass  # Fallback to checking /v1/models list on connection errors
            except Exception:  # Catch JSON errors etc.
                pass  # Fallback

        # Fallback or if using /v1: Check the /v1/models list (only shows loaded models)
        try:
            url = (
                f"{self.base}/v1/models"  # Always check /v1/models as fallback/default
            )
            r = requests.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            loaded_models = r.json().get("data", [])
            return any(m.get("id") == self.raw_id for m in loaded_models)
        except Exception as e:
            _debug(
                f"LMSTUDIO DEBUG: Could not check loaded models via /v1/models: {e}"
            )
            return False  # Assume not loaded if check fails

    def _attempt_load_model(self) -> bool:
        """Load the model through LM Studio's synchronous REST endpoint."""
        _debug(
            f"LMSTUDIO INFO: Model '{self.raw_id}' not loaded. Attempting to load..."
        )

        try:
            response = requests.post(
                f"{self.base}/api/v1/models/load",
                json={"model": self.raw_id},
                timeout=TIMEOUT,
            )
            response.raise_for_status()
            result = response.json()
            if result.get("status") != "loaded":
                raise ValueError(f"Unexpected load response: {result}")

            duration = result.get("load_time_seconds")
            duration_text = (
                f" in {duration:.3f}s" if isinstance(duration, int | float) else ""
            )
            _debug(
                f"LMSTUDIO INFO: Model '{self.raw_id}' loaded{duration_text} as instance '{result.get('instance_id')}'."
            )
            return True
        except requests.RequestException as e:
            message = str(e)
            if e.response is not None:
                try:
                    error = e.response.json().get("error", {})
                    message = error.get("message", message)
                except (ValueError, AttributeError):
                    pass
            print(f"LMSTUDIO ERROR: Failed to load model: {message}", file=sys.stderr)
            return False
        except (ValueError, TypeError) as e:
            print(f"LMSTUDIO ERROR: Failed to load model: {e}", file=sys.stderr)
            return False

    # --------------------------------------------------------------------- #
    #  Prompt helpers                                                       #
    # --------------------------------------------------------------------- #
    def _encode_tools(self, tools: list[llm.Tool]) -> list[dict]:
        """Convert llm.Tool objects to LM Studio tools format."""
        encoded_tools = []
        for tool in tools:
            encoded_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    },
                }
            )
        return encoded_tools

    def _encode_tool_results(self, tool_results: list[llm.ToolResult]) -> list[dict]:
        """Convert llm.ToolResult objects to LM Studio message format."""
        messages = []
        for result in tool_results:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": result.tool_call_id,
                    "content": result.output,
                }
            )
        return messages

    def _build_messages(self, prompt: llm.Prompt, conversation=None) -> list[dict]:
        """Translate LLM's canonical structured chain to Chat Completions.

        Since LLM 0.32, ``prompt.messages`` already contains the complete
        history. Walking ``conversation.responses`` as well would duplicate
        that history, including the system prompt on chained tool calls.
        """
        messages: list[dict] = []
        current_system: str | None = None
        attachments = []

        for message in prompt.messages:
            text_bits = []
            reasoning_bits = []
            image_parts = []
            tool_calls = []
            tool_results = []
            had_attachment = False

            for part in message.parts:
                if isinstance(part, TextPart):
                    text_bits.append(part.text)
                elif isinstance(part, ReasoningPart):
                    if part.text and not part.redacted:
                        reasoning_bits.append(part.text)
                elif isinstance(part, AttachmentPart) and part.attachment:
                    had_attachment = True
                    attachments.append(part.attachment)
                    image_parts.extend(self._encode_attachment(part.attachment))
                elif isinstance(part, ToolCallPart):
                    tool_calls.append(
                        {
                            "id": part.tool_call_id,
                            "type": "function",
                            "function": {
                                "name": part.name,
                                "arguments": json.dumps(part.arguments),
                            },
                        }
                    )
                elif isinstance(part, ToolResultPart):
                    tool_results.append(
                        {
                            "role": "tool",
                            "tool_call_id": part.tool_call_id,
                            "content": part.output,
                        }
                    )

            if message.role == "tool":
                messages.extend(tool_results)
                continue

            text = "".join(text_bits)
            if message.role == "system":
                # OpenAI-compatible endpoints accept one active system prompt.
                # Repeated unchanged system messages can occur in explicit
                # histories, so avoid sending them more than once.
                if text == current_system:
                    continue
                current_system = text

            if image_parts:
                content = []
                if text:
                    content.append({"type": "text", "text": text})
                content.extend(image_parts)
            else:
                content = text or None

            entry = {"role": message.role, "content": content}
            if reasoning_bits and message.role == "assistant":
                # LM Studio exposes reasoning from OpenAI-compatible models in
                # this field and accepts it again in conversation history.
                entry["reasoning_content"] = "".join(reasoning_bits)
            if tool_calls:
                entry["tool_calls"] = tool_calls
                if not text and not image_parts:
                    entry["content"] = None
            elif had_attachment and message.role == "user" and entry["content"] is None:
                # Keep a user turn when its only attachment could not be
                # encoded, rather than sending an entirely empty chain.
                entry["content"] = ""

            # An assistant message containing only tool calls is meaningful;
            # empty user/system messages are not.
            if (
                entry["content"] is None
                and not tool_calls
                and not (reasoning_bits and message.role == "assistant")
            ):
                continue
            messages.append(entry)

        self._warn_for_unsupported_attachments(attachments)
        return messages

    def _warn_for_unsupported_attachments(self, attachments) -> None:
        if self.supports_images or not attachments:
            return
        for attachment in attachments:
            try:
                if attachment.resolve_type() not in self.attachment_types:
                    continue
            except Exception:
                pass
            print(
                f"LMSTUDIO WARN: Attachments provided, but the selected model '{self.model_id}' "
                f"may not support images (supports_images={self.supports_images}). Image attachments will likely be ignored by the model.",
                file=sys.stderr,
            )
            return

    def _encode_attachment(self, attachment: llm.Attachment) -> list[dict]:
        """Encode one image attachment as an OpenAI image_url content part."""
        if not self.supports_images:
            _debug(
                f"LMSTUDIO DEBUG: Model {self.model_id} does not support images, but attachment {attachment.path or attachment.url or 'content'} was provided. Ignoring."
            )
            return []
        try:
            resolved_type = attachment.resolve_type()
            if resolved_type not in self.attachment_types:
                _debug(
                    f"LMSTUDIO DEBUG: Attachment type {resolved_type} not in model's supported image types. Skipping {attachment.path or attachment.url or 'content'}."
                )
                return []
            base64_content = attachment.base64_content()
            data_uri = f"data:{resolved_type};base64,{base64_content}"
            _debug(
                f"LMSTUDIO DEBUG: Encoded image attachment: {attachment.path or attachment.url or 'content'} as {resolved_type}."
            )
            return [{"type": "image_url", "image_url": {"url": data_uri}}]
        except Exception as e:
            print(
                f"LMSTUDIO WARN: Could not process attachment {attachment.path or attachment.url or 'content'}: {e}. Skipping.",
                file=sys.stderr,
            )
            return []

    def _encode_attachments(self, prompt: llm.Prompt) -> list[dict]:
        """Backward-compatible helper for encoding prompt attachments."""
        encoded_attachments = []
        for attachment in prompt.attachments:
            encoded_attachments.extend(self._encode_attachment(attachment))
        return encoded_attachments

    def _tool_call_from_data(self, tool_call_data: dict) -> tuple[llm.ToolCall, str]:
        function_data = tool_call_data.get("function", {})
        arguments_json = function_data.get("arguments") or "{}"
        return (
            llm.ToolCall(
                name=function_data.get("name", ""),
                arguments=json.loads(arguments_json),
                tool_call_id=tool_call_data.get("id") or f"tc_{uuid.uuid4().hex}",
            ),
            arguments_json,
        )

    def _record_tool_call(self, response, tool_call_data: dict) -> list[StreamEvent]:
        tool_call, arguments_json = self._tool_call_from_data(tool_call_data)
        response.add_tool_call(tool_call)
        return [
            StreamEvent(
                type="tool_call_name",
                chunk=tool_call.name,
                tool_call_id=tool_call.tool_call_id,
            ),
            StreamEvent(
                type="tool_call_args",
                chunk=arguments_json,
                tool_call_id=tool_call.tool_call_id,
            ),
        ]

    def _set_response_metadata(self, response, payload: dict) -> None:
        response.response_json = payload
        resolved_model = payload.get("model")
        if not resolved_model and payload.get("chunks"):
            resolved_model = next(
                (
                    chunk.get("model")
                    for chunk in reversed(payload["chunks"])
                    if chunk.get("model")
                ),
                None,
            )
        if resolved_model:
            response.set_resolved_model(resolved_model)

    def _set_usage(self, response, usage: dict | None) -> None:
        if not usage:
            return
        details = {
            key: value
            for key, value in usage.items()
            if key not in {"prompt_tokens", "completion_tokens", "total_tokens"}
        }
        response.set_usage(
            input=usage.get("prompt_tokens"),
            output=usage.get("completion_tokens"),
            details=details or None,
        )

    def _process_non_streaming_response(
        self,
        response,
        payload: dict[str, Any],
    ) -> Iterator[StreamEvent]:
        self._set_response_metadata(response, payload)
        choice = payload.get("choices", [{}])[0]
        message = choice.get("message", {})

        reasoning = message.get("reasoning_content") or message.get("reasoning")
        if reasoning:
            yield StreamEvent(type="reasoning", chunk=reasoning)

        for tool_call_data in message.get("tool_calls") or []:
            try:
                yield from self._record_tool_call(response, tool_call_data)
            except (json.JSONDecodeError, TypeError) as e:
                _debug(f"LMSTUDIO DEBUG: Error processing tool call: {e}")

        if message.get("content"):
            yield StreamEvent(type="text", chunk=message["content"])

        self._set_usage(response, payload.get("usage"))

    def _process_stream_line(
        self,
        line: str,
        state: StreamState,
    ) -> Iterator[StreamEvent]:
        if not line or line == "data: [DONE]" or not line.startswith("data:"):
            return

        chunk_data = line[5:].strip()
        if not chunk_data:
            return
        try:
            chunk = json.loads(chunk_data)
        except json.JSONDecodeError as e:
            _debug(f"LMSTUDIO DEBUG: Ignoring malformed stream JSON: {e}")
            return
        if not isinstance(chunk, dict):
            _debug("LMSTUDIO DEBUG: Ignoring non-object stream chunk")
            return

        state.chunks.append(chunk)
        usage = chunk.get("usage")
        if isinstance(usage, dict):
            state.usage = usage
        elif usage is not None:
            _debug("LMSTUDIO DEBUG: Ignoring malformed stream usage")

        choices = chunk.get("choices")
        if not choices:
            return
        if not isinstance(choices, list) or not isinstance(choices[0], dict):
            _debug("LMSTUDIO DEBUG: Ignoring malformed stream choices")
            return

        delta = choices[0].get("delta", {})
        if not isinstance(delta, dict):
            _debug("LMSTUDIO DEBUG: Ignoring malformed stream delta")
            return

        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            yield StreamEvent(type="reasoning", chunk=reasoning)
        elif reasoning is not None and not isinstance(reasoning, str):
            _debug("LMSTUDIO DEBUG: Ignoring malformed reasoning fragment")

        token = delta.get("content")
        if isinstance(token, str) and token:
            yield StreamEvent(type="text", chunk=token)
        elif token is not None and not isinstance(token, str):
            _debug("LMSTUDIO DEBUG: Ignoring malformed text fragment")

        tool_call_deltas = delta.get("tool_calls") or []
        if not isinstance(tool_call_deltas, list):
            _debug("LMSTUDIO DEBUG: Ignoring malformed tool-call list")
            return
        for tool_call_delta in tool_call_deltas:
            try:
                self._apply_tool_call_delta(tool_call_delta, state)
            except (TypeError, ValueError) as e:
                _debug(f"LMSTUDIO DEBUG: Ignoring malformed tool-call delta: {e}")

    def _apply_tool_call_delta(
        self,
        tool_call_delta: Any,
        state: StreamState,
    ) -> None:
        if not isinstance(tool_call_delta, dict):
            raise TypeError("tool-call delta must be an object")

        index = tool_call_delta.get("index")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError("tool-call index must be a non-negative integer")

        tool_call_id = tool_call_delta.get("id")
        if tool_call_id is None:
            tool_call_id = ""
        function_delta = tool_call_delta.get("function")
        if function_delta is None:
            function_delta = {}
        if not isinstance(function_delta, dict):
            raise TypeError("tool-call function must be an object")
        name = function_delta.get("name")
        if name is None:
            name = ""
        arguments = function_delta.get("arguments")
        if arguments is None:
            arguments = ""
        if not all(
            isinstance(value, str) for value in (tool_call_id, name, arguments)
        ):
            raise TypeError("tool-call fragments must be strings")

        while len(state.tool_calls) <= index:
            state.tool_calls.append(
                {
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                }
            )

        tool_call = state.tool_calls[index]
        tool_call["id"] += tool_call_id
        tool_call["function"]["name"] += name
        tool_call["function"]["arguments"] += arguments

    def _finalize_stream(
        self,
        response,
        state: StreamState,
    ) -> Iterator[StreamEvent]:
        self._set_response_metadata(response, {"chunks": state.chunks})
        self._set_usage(response, state.usage)
        for tool_call_data in state.tool_calls:
            try:
                yield from self._record_tool_call(response, tool_call_data)
            except (json.JSONDecodeError, TypeError) as e:
                _debug(f"LMSTUDIO DEBUG: Error processing tool call: {e}")


class LMStudioModel(LMStudioBaseModel, llm.Model):
    """Chat/completion model class."""

    # --------------------------------------------------------------------- #
    #  Execute                                                              #
    # --------------------------------------------------------------------- #
    def execute(
        self,
        prompt: llm.Prompt,
        stream: bool,
        response: llm.Response,
        conversation=None,
    ) -> Iterator[str | StreamEvent]:
        # --- Auto-loading Logic ---
        if not self._is_model_loaded():
            if not self._attempt_load_model():
                raise llm.ModelError(
                    f"Failed to load model '{self.raw_id}' through the LM Studio API."
                )
            else:
                time.sleep(1)  # Add a small delay after successful load confirmation
        # --- End Auto-loading Logic ---

        request = self._prepare_chat_request(prompt, stream, conversation)
        stream = request.stream

        # --- Execute API Call --- #
        try:
            r = requests.post(
                request.url,
                json=request.payload,
                stream=request.stream,
                timeout=request.timeout,
            )
            r.raise_for_status()
        except requests.exceptions.Timeout:
            # Specific handling for timeout error
            if hasattr(prompt, "tools") and prompt.tools:
                raise llm.ModelError(
                    f"LM Studio request with tools timed out after {request.timeout} seconds. "
                    f"This model may not properly support tools, or the request is taking too long. "
                    f"Try increasing LMSTUDIO_TIMEOUT environment variable or using a different model."
                )
            else:
                raise llm.ModelError(
                    f"LM Studio request timed out after {request.timeout} seconds. "
                    f"Try increasing LMSTUDIO_TIMEOUT environment variable or using a faster model."
                )
        except requests.RequestException as e:
            is_model_not_found = False
            try:
                if e.response is not None:
                    err_data = e.response.json()
                    if (
                        isinstance(err_data, dict)
                        and err_data.get("error", {}).get("code") == "model_not_found"
                    ):
                        is_model_not_found = True
            except Exception:
                pass  # Ignore JSON parsing errors here

            if is_model_not_found:
                raise llm.ModelError(
                    f"Model '{self.raw_id}' not found by LM Studio server at {request.url}, even after attempting auto-load. Is it correctly specified and loadable?"
                )
            else:
                raise llm.ModelError(f"LM Studio request failed: {e}")
        # --- End Execute API Call --- #

        # --- Process Response --- #
        if stream:
            state = StreamState()
            for line in r.iter_lines():
                try:
                    decoded_line = line.decode("utf-8")
                except UnicodeDecodeError as e:
                    _debug(
                        f"LMSTUDIO DEBUG: Ignoring invalid UTF-8 stream line: {e}"
                    )
                    continue
                yield from self._process_stream_line(decoded_line, state)
            yield from self._finalize_stream(response, state)

        else:  # Non-streaming
            try:
                raw_text = r.text  # Get raw text first
                res = r.json()
            except json.JSONDecodeError as e:
                print(
                    f"LMSTUDIO ERROR: Failed to decode JSON response: {e}",
                    file=sys.stderr,
                )
                _debug(f"LMSTUDIO DEBUG: Failing raw text was: {raw_text}")
                raise llm.ModelError("Failed to decode JSON response from LM Studio.")

            yield from self._process_non_streaming_response(response, res)

        # --- End Process Response --- #


# ------------------------  Async Model  ------------------------------------ #
class LMStudioAsyncModel(LMStudioBaseModel, llm.AsyncModel):
    """Async version of the chat/completion model class."""

    async def execute(
        self,
        prompt: llm.Prompt,
        stream: bool,
        response: llm.AsyncResponse,
        conversation: llm.AsyncConversation | None,
    ) -> AsyncGenerator[str | StreamEvent, None]:
        # --- Auto-loading Logic (using sync helper) ---
        if not self._is_model_loaded():
            if not self._attempt_load_model():
                raise llm.ModelError(
                    f"Failed to load model '{self.raw_id}' through the LM Studio API."
                )
            else:
                # No async sleep needed here as load itself is sync
                pass
        # --- End Auto-loading Logic ---

        request = self._prepare_chat_request(prompt, stream, conversation)
        stream = request.stream

        # --- Execute API Call (Async) ---
        try:
            async with httpx.AsyncClient(timeout=request.timeout) as client:
                if request.stream:
                    async with client.stream(
                        "POST", request.url, json=request.payload
                    ) as r:
                        r.raise_for_status()
                        state = StreamState()
                        async for line in r.aiter_lines():
                            for event in self._process_stream_line(line, state):
                                yield event
                        for event in self._finalize_stream(response, state):
                            yield event

                else:  # Non-streaming async
                    r = await client.post(request.url, json=request.payload)
                    r.raise_for_status()
                    try:
                        raw_text = r.text
                        res = r.json()
                    except json.JSONDecodeError as e:
                        print(
                            f"LMSTUDIO ERROR: Failed to decode JSON response: {e}",
                            file=sys.stderr,
                        )
                        _debug(
                            f"LMSTUDIO DEBUG: Failing raw text was: {raw_text}"
                        )
                        raise llm.ModelError(
                            "Failed to decode JSON response from LM Studio."
                        )

                    for event in self._process_non_streaming_response(response, res):
                        yield event

        except httpx.TimeoutException:
            if hasattr(prompt, "tools") and prompt.tools:
                raise llm.ModelError(
                    f"LM Studio async request with tools timed out after {request.timeout} seconds. "
                    f"This model may not properly support tools, or the request is taking too long. "
                    f"Try increasing LMSTUDIO_TIMEOUT environment variable or using a different model."
                )
            else:
                raise llm.ModelError(
                    f"LM Studio async request timed out after {request.timeout} seconds. "
                    f"Try increasing LMSTUDIO_TIMEOUT environment variable."
                )
        except httpx.RequestError as e:
            # Basic error handling, could be refined like the sync version
            raise llm.ModelError(f"LM Studio async request failed: {e}")


# ------------------------  Embedding  ------------------------------------- #
class LMStudioEmbeddingModel(llm.EmbeddingModel):
    def __init__(self, model_id: str, base_url: str, raw_id: str, api_path_prefix: str):
        self.model_id = model_id
        self.raw_id = raw_id
        self.base = base_url
        self.api_path_prefix = api_path_prefix

    def embed_batch(self, items: Iterable[str | bytes]) -> Iterator[list[float]]:
        try:
            r = requests.post(
                f"{self.base}{self.api_path_prefix}/embeddings",
                json={"model": self.raw_id, "input": items},
                timeout=TIMEOUT,
            )
            r.raise_for_status()
            data = r.json()

            return (cast(list[float], item["embedding"]) for item in data["data"])
        except requests.RequestException as e:
            raise llm.ModelError(f"LM Studio embeddings request failed: {e}") from e
        except (KeyError, TypeError) as e:
            raise llm.ModelError(f"Unexpected embeddings response: {e}") from e
