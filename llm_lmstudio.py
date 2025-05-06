"""
llm‑lmstudio — enhanced plugin
Requires llm >= 0.23 (attachments API) and LM Studio ≥ 0.3.6 for /api/v0/models.
"""
from __future__ import annotations
import base64, json, mimetypes, os, pathlib, sys, time, subprocess
from typing import List, Optional, Dict, Any, AsyncGenerator
from urllib.parse import urlparse

import requests
import httpx
import llm
from pydantic import Field

# --------------------------------------------------------------------------- #
#  Configuration                                                              #
# --------------------------------------------------------------------------- #
raw = (os.getenv("LMSTUDIO_API_BASE")       # singular, supports comma-separated list
       or "http://localhost:1234")             # hard default
SERVER_LIST = [u.strip().rstrip("/") for u in raw.split(",") if u.strip()]
TIMEOUT = float(os.getenv("LMSTUDIO_TIMEOUT", 4))

# --------------------------------------------------------------------------- #
#  Internal helpers                                                           #
# --------------------------------------------------------------------------- #
_cache: Dict[str, tuple[List[Dict[str, Any]], str]] = {}
_errors: Dict[str, Exception] = {}

def _fetch_models(base: str) -> tuple[List[Dict[str, Any]], str]:
    """Return cached metadata and API path prefix for one LM Studio server."""
    if base in _cache:
        return _cache[base]
    try:
        # Prefer the richer metadata endpoint
        api_path = "/api/v0"
        if os.getenv("LLM_LMSTUDIO_DEBUG") == "1":
            print(f"LMSTUDIO DEBUG: Fetching models from {base}{api_path}/models", file=sys.stderr)
        r = requests.get(f"{base}{api_path}/models", timeout=TIMEOUT)
        if r.status_code == 404:  # Older LM Studio → fall back
            api_path = "/v1"
            if os.getenv("LLM_LMSTUDIO_DEBUG") == "1":
                print(f"LMSTUDIO DEBUG: {base}/api/v0/models not found, falling back to {base}{api_path}/models", file=sys.stderr)
            r = requests.get(f"{base}{api_path}/models", timeout=TIMEOUT)
            r.raise_for_status()
            data = r.json().get("data", [])
            if os.getenv("LLM_LMSTUDIO_DEBUG") == "1":
                print(f"LMSTUDIO DEBUG: Received data from /v1 endpoint for {base}: {data}", file=sys.stderr) # Print all v1 data
            # v1 has no 'type'; assume plain LLM or infer from ID
            meta = []
            for m_data in data:
                m_id = m_data["id"] if isinstance(m_data, dict) else m_data
                m_type = "embeddings" if "embed" in m_id.lower() else "llm"
                # V1 doesn't reliably tell us VLM status
                current_model_meta = {"id": m_id, "type": m_type, "vision": False}
                if os.getenv("LLM_LMSTUDIO_DEBUG") == "1":
                    print(f"LMSTUDIO DEBUG: Processed /v1 model data for {base}: ID={m_id}, Inferred Type={m_type}, Vision=False", file=sys.stderr)
                meta.append(current_model_meta)
        else:
            r.raise_for_status()
            meta = r.json().get("data", [])
            if os.getenv("LLM_LMSTUDIO_DEBUG") == "1":
                print(f"LMSTUDIO DEBUG: Received full metadata from /api/v0 for {base}:", file=sys.stderr)
                for m_debug in meta:
                    print(f"  LMSTUDIO DEBUG: Model ID: {m_debug.get('id')}, Type: {m_debug.get('type')}, Original Vision: {m_debug.get('vision')}, Path: {m_debug.get('path')}, Publisher: {m_debug.get('publisher')}, Architecture: {m_debug.get('architecture')}, Quantization: {m_debug.get('quantization')}", file=sys.stderr) # Added more fields

            # Add 'vision' flag for /api/v0 models for clarity
            for m in meta:
                # Explicitly check if type is 'vlm'
                is_vlm_type = m.get("type") == "vlm"
                # Check if 'vision' key exists and is true
                has_vision_flag = m.get("vision") is True # Explicitly check for True
                
                m['vision'] = is_vlm_type or has_vision_flag # Set our 'vision' flag based on these
                if os.getenv("LLM_LMSTUDIO_DEBUG") == "1":
                    print(f"  LMSTUDIO DEBUG: For model {m.get('id')}: Original type='{m.get('type')}', original vision_key_present_and_true='{has_vision_flag}', calculated plugin vision_support='{m['vision']}'", file=sys.stderr)


        _cache[base] = (meta, api_path)
        return meta, api_path
    except Exception as e:
        _errors[base] = e
        return [], "" # Return empty list and empty path on error

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
        if not models and not api_path: # Skip if fetch failed completely
             continue
        for m in models:
            if m.get("type") == "embeddings":
                continue  # handled in embedding hook

            raw_id = m['id']
            if single_server:
                model_id = raw_id
            else:
                model_id = f"lmstudio@{_host_tag(base)}/{raw_id}"

            # Check if model is loaded (only reliable via /api/v0)
            is_loaded = m.get("state") == "loaded"
            display_model_id = f"{model_id} 🟢" if is_loaded else model_id

            # Use the 'vision' flag that was calculated and refined by _fetch_models
            supports_images_flag = m.get('vision', False)

            # Modify the actual model_id that will be registered
            # final_model_id_for_registration = model_id # This was the old approach
            # The actual model_id should remain clean for lookup.
            
            display_suffix_parts = []
            if is_loaded:
                display_suffix_parts.append("●") # Black Circle for loaded
            if supports_images_flag:
                display_suffix_parts.append("👁")  # Eye for vision (U+1F441)
            # Assuming schema support for these models, as it's a general capability of the endpoint
            display_suffix_parts.append("⚒")  # Hammer and Pick for schema/tools (U+2692)

            display_suffix = ""
            if display_suffix_parts:
                display_suffix = " " + " ".join(display_suffix_parts) # Join with spaces, add leading space

            if os.getenv("LLM_LMSTUDIO_DEBUG") == "1":
                print(f"LMSTUDIO DEBUG [register_models]: Base model_id: '{model_id}', Calculated display_suffix: '{display_suffix}'", file=sys.stderr)
                print(f"LMSTUDIO DEBUG [register_models]: For {raw_id}, passing model_id='{model_id}', supports_images={supports_images_flag} to constructor.", file=sys.stderr)
            
            current_metadata = {
                "publisher": m.get("publisher"),
                "arch": m.get("arch"), # Note: LM Studio API docs say 'architecture' but example shows 'arch'
                "quantization": m.get("quantization"),
                "max_context_length": m.get("max_context_length"),
                "state": m.get("state", "unknown"), # Default to unknown if not available
                "api_path": api_path,    # Store the API path used
                "base_url": base,        # Store the base URL
                "vision": supports_images_flag, # Ensure our calculated flag is in metadata
                "raw_lmstudio_type": m.get("type") # Store original type for debugging/inspection
            }
            # Filter out None values for cleaner inspection, but keep 'vision' as it's boolean
            current_metadata = {k: v for k, v in current_metadata.items() if v is not None or k == 'vision'}

            register(
                LMStudioModel(model_id, base, raw_id, api_path, supports_images=supports_images_flag, metadata=current_metadata, display_suffix=display_suffix),
                LMStudioAsyncModel(model_id, base, raw_id, api_path, supports_images=supports_images_flag, metadata=current_metadata, display_suffix=display_suffix)
            )
    if _errors:
        print(
            "Warning: Some LM Studio servers were unreachable:\\n  "
            + "\\n  ".join(f"{k}: {v}" for k, v in _errors.items()),
            file=sys.stderr,
        )

@llm.hookimpl
def register_embedding_models(register):
    single_server = len(SERVER_LIST) == 1
    for base in SERVER_LIST:
        models, api_path = _fetch_models(base)
        if not models and not api_path: # Skip if fetch failed completely
            continue
        for m in models:
            if m.get("type") == "embeddings":
                raw_id = m['id']
                if single_server:
                    model_id = raw_id
                else:
                    model_id = f"lmstudio@{_host_tag(base)}/{raw_id}"
                register(LMStudioEmbeddingModel(model_id, base, raw_id, api_path))

# --------------------------------------------------------------------------- #
#  Model classes                                                              #
# --------------------------------------------------------------------------- #
class LMStudioBaseModel(llm.Model):
    """Base class for common LMStudio model attributes."""
    can_stream: bool = True
    attachment_types = { # Task 1.1
        "image/png",
        "image/jpeg",
        "image/gif",
        "image/webp",
    }

class LMStudioModel(LMStudioBaseModel):
    """Chat/completion model class."""

    class Options(llm.Options):
        temperature: Optional[float] = Field(None, description="Sampling temperature")
        top_p: Optional[float] = Field(None, description="Nucleus sampling")
        max_tokens: Optional[int] = Field(None, description="Maximum tokens")
        stop: Optional[List[str]] = Field(None, description="Stop sequences")

    def __init__(self, model_id: str, base_url: str, raw_id: str, api_path_prefix: str, *, supports_images: bool = False, metadata: dict = None, display_suffix: str = ""):
        self.model_id = model_id # This is the clean ID for lookup
        self.raw_id = raw_id
        self.base = base_url
        self.api_path_prefix = api_path_prefix
        self.supports_images = supports_images
        self.metadata = metadata or {}
        self.supports_schema = True 
        self.display_suffix = display_suffix # Store the suffix for __str__

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
                elif r.status_code == 404: # Model exists but endpoint doesn't? Unlikely but handle
                    pass # Fallback to checking /v1/models list
                else:
                    r.raise_for_status() # Raise other errors
            except requests.RequestException:
                 pass # Fallback to checking /v1/models list on connection errors
            except Exception: # Catch JSON errors etc.
                 pass # Fallback

        # Fallback or if using /v1: Check the /v1/models list (only shows loaded models)
        try:
            url = f"{self.base}/v1/models" # Always check /v1/models as fallback/default
            r = requests.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            loaded_models = r.json().get("data", [])
            return any(m.get("id") == self.raw_id for m in loaded_models)
        except Exception as e:
            print(f"LMSTUDIO WARN: Could not check loaded models via /v1/models: {e}", file=sys.stderr)
            return False # Assume not loaded if check fails

    def _attempt_load_model(self):
        """Attempt to load the model using 'lms load' and show simple progress."""
        debug_enabled = os.getenv("LLM_LMSTUDIO_DEBUG") == "1"

        if debug_enabled:
            print(f"LMSTUDIO INFO: Model '{self.raw_id}' not loaded. Attempting to load...", file=sys.stderr)
        else:
            # Show minimal non-debug message
            print(f"\rLoading model '{self.raw_id}'...", end='', file=sys.stderr)

        try:
            process = subprocess.Popen(
                ["lms", "load", self.raw_id],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True, # Decode output as text
                bufsize=1 # Line buffered
            )

            # --- Progress Display (only if not in debug mode) ---
            if not debug_enabled and process.stdout:
                for line in iter(process.stdout.readline, ''):
                    stripped_line = line.strip()
                    if stripped_line and '%' in stripped_line:
                        # Extract just the progress bar part if possible, else show line
                        progress_part = stripped_line
                        try:
                            # Try to extract just the bar and percentage
                            bar_start = stripped_line.find('[')
                            if bar_start != -1:
                                progress_part = stripped_line[bar_start:]
                        except Exception:
                            pass # Keep original line if parsing fails
                        print(f"\rLoading: {progress_part:<80}", end='', file=sys.stderr)
                    elif stripped_line and debug_enabled: # Print non-progress stdout only in debug
                         print(f"\nLMSTUDIO DEBUG [stdout]: {stripped_line}", file=sys.stderr)
            # --- End Progress Display ---

            # Capture stderr
            stderr_output = ""
            if process.stderr:
                stderr_output = process.stderr.read()
                if debug_enabled and stderr_output.strip(): # Show stderr only in debug
                    print(f"\nLMSTUDIO DEBUG [stderr]:\n{stderr_output.strip()}", file=sys.stderr)

            # Clear the loading/progress line
            print(f"\r{' ':<90}\r", end='', file=sys.stderr)

            process.wait(timeout=1) # Short wait for final exit code
            if debug_enabled:
                print(f"LMSTUDIO DEBUG: 'lms load {self.raw_id}' command finished with code {process.returncode}.", file=sys.stderr)

            if process.returncode != 0:
                 # Always print this error
                 print(f"LMSTUDIO ERROR: 'lms load {self.raw_id}' failed (code {process.returncode}).", file=sys.stderr)
                 if stderr_output.strip(): # Show stderr on error regardless of debug flag
                    print(f"LMSTUDIO ERROR [stderr]:\n{stderr_output.strip()}", file=sys.stderr)
                 return False

            # Poll to confirm API status
            max_wait_seconds = 30
            poll_interval_seconds = 1
            start_time = time.time()
            while time.time() - start_time < max_wait_seconds:
                if self._is_model_loaded():
                    if debug_enabled:
                        print(f"LMSTUDIO INFO: Model '{self.raw_id}' successfully loaded (confirmed via API).", file=sys.stderr)
                    return True
                time.sleep(poll_interval_seconds)

            # Always print this error
            print(f"LMSTUDIO ERROR: Model '{self.raw_id}' not detected as loaded via API within {max_wait_seconds} seconds after 'lms load' completed.", file=sys.stderr)
            return False

        except FileNotFoundError:
             print(f"\r{' ':<90}\r", end='', file=sys.stderr) # Clear potential progress line
             # Always print this error
             print(f"LMSTUDIO ERROR: 'lms' command not found. Cannot auto-load model.", file=sys.stderr)
             print(f"Please ensure LM Studio CLI is installed and in your PATH, or load the model manually.", file=sys.stderr)
             return False
        except subprocess.TimeoutExpired:
            print(f"\r{' ':<90}\r", end='', file=sys.stderr) # Clear potential progress line
            # Always print this error
            print(f"LMSTUDIO ERROR: Waiting for 'lms load {self.raw_id}' command timed out.", file=sys.stderr)
            try: process.kill()
            except Exception: pass
            return False
        except Exception as e:
            print(f"\r{' ':<90}\r", end='', file=sys.stderr) # Clear potential progress line
            # Always print this error
            print(f"LMSTUDIO ERROR: An unexpected error occurred while trying to load model: {e}", file=sys.stderr)
            return False

    # --------------------------------------------------------------------- #
    #  Prompt helpers                                                       #
    # --------------------------------------------------------------------- #
    def _encode_attachments(self, prompt: llm.Prompt) -> List[dict]:
        """Encode attachments from the prompt using llm.Attachment API.""" # Task 1.2
        encoded_attachments = []
        if not prompt.attachments:
            return encoded_attachments

        for attachment in prompt.attachments:
            # Only process attachments if the model supports images
            # and the attachment type is one of the supported image types.
            # llm CLI should have already filtered by attachment_types,
            # but this is an additional safeguard.
            if self.supports_images:
                try:
                    resolved_type = attachment.resolve_type()
                    if resolved_type in self.attachment_types:
                        base64_content = attachment.base64_content()
                        data_uri = f"data:{resolved_type};base64,{base64_content}"
                        encoded_attachments.append(
                            {"type": "image_url", "image_url": {"url": data_uri}}
                        )
                        if os.getenv("LLM_LMSTUDIO_DEBUG") == "1":
                            print(f"LMSTUDIO DEBUG: Encoded image attachment: {attachment.path or attachment.url or 'content'} as {resolved_type}.", file=sys.stderr)
                    elif os.getenv("LLM_LMSTUDIO_DEBUG") == "1":
                         print(f"LMSTUDIO DEBUG: Attachment type {resolved_type} not in model's supported image types. Skipping {attachment.path or attachment.url or 'content'}.", file=sys.stderr)

                except Exception as e:
                    print(f"LMSTUDIO WARN: Could not process attachment {attachment.path or attachment.url or 'content'}: {e}. Skipping.", file=sys.stderr)
            elif os.getenv("LLM_LMSTUDIO_DEBUG") == "1": # Model does not support images but attachments present
                print(f"LMSTUDIO DEBUG: Model {self.model_id} does not support images, but attachment {attachment.path or attachment.url or 'content'} was provided. Ignoring.", file=sys.stderr)
        return encoded_attachments

    def _build_messages(self, prompt: llm.Prompt, conversation) -> List[dict]:
        msgs: List[dict] = []
        if prompt.system:
            msgs.append({"role": "system", "content": prompt.system})

        # Task 2.1: Warn if attachments are present but model doesn't support images
        if prompt.attachments and not self.supports_images:
            # Check if any of the attachments are of a type the model *would* process if it supported images
            # This avoids warning if only non-image attachments are present (future-proofing)
            has_potential_image_attachment = False
            for attachment in prompt.attachments:
                try:
                    resolved_type = attachment.resolve_type()
                    if resolved_type in self.attachment_types: # self.attachment_types contains image MIME types
                        has_potential_image_attachment = True
                        break
                except Exception:
                    # If type resolution fails, assume it might have been an image for warning purposes
                    has_potential_image_attachment = True
                    break
            
            if has_potential_image_attachment:
                print(
                    f"LMSTUDIO WARN: Attachments provided, but the selected model '{self.model_id}' "
                    f"may not support images (supports_images={self.supports_images}). Image attachments will likely be ignored by the model.",
                    file=sys.stderr
                )

        if conversation:
            for prev in conversation.responses:
                # Add system prompt from previous turn if it exists
                if hasattr(prev.prompt, 'system') and prev.prompt.system:
                    # Avoid duplicate system message if the immediate previous message was already system.
                    # This can happen if a system message was the very first in conversation.
                    if not msgs or msgs[-1]['role'] != 'system' or msgs[-1]['content'] != prev.prompt.system:
                        msgs.append({"role": "system", "content": prev.prompt.system})
                
                prev_prompt_text = prev.prompt.prompt
                prev_encoded_attachments = []

                # Check if the previous prompt object has attachments and encode them
                # This requires that llm.Response objects store attachments correctly
                # or that we can access the original llm.Prompt for prev.prompt.
                # Assuming prev.prompt has an `attachments` attribute similar to the current prompt.
                if hasattr(prev.prompt, 'attachments') and prev.prompt.attachments:
                    # Temporarily create attachment list for the previous prompt
                    # We need to ensure that `self._encode_attachments` can be called here.
                    # This implies `prev.prompt` should be an `llm.Prompt` like object or we adapt.
                    # For now, this assumes `prev.prompt` is compatible.
                    # A more robust way might be to store encoded attachments on the response object itself if llm allows.
                    # However, the llm docs example for OpenAI shows re-encoding like this:
                    prev_encoded_attachments = self._encode_attachments(prev.prompt)

                if prev_encoded_attachments:
                    current_content_parts = []
                    if prev_prompt_text:
                        current_content_parts.append({"type": "text", "text": prev_prompt_text})
                    current_content_parts.extend(prev_encoded_attachments)
                    msgs.append({"role": "user", "content": current_content_parts})
                elif prev_prompt_text:
                    msgs.append({"role": "user", "content": prev_prompt_text})
                else: # Should not happen if prompt always has text or attachment
                    msgs.append({"role": "user", "content": ""}) # Empty content if no text/attachments

                msgs.append({"role": "assistant", "content": prev.text_or_raise()})

        # current turn
        current_turn_content_parts = []
        if prompt.prompt: # Add text part if it exists
            current_turn_content_parts.append({"type": "text", "text": prompt.prompt})
        
        img_parts = self._encode_attachments(prompt)
        current_turn_content_parts.extend(img_parts)

        if not current_turn_content_parts:
            # This case means no text prompt AND no processable image attachments.
            # Sending an empty content list might be an error for the API.
            # For now, send an empty string as content, which is likely to be ignored or handled by the model.
            # Or, we could raise an error if a prompt is truly empty.
            # The llm CLI usually ensures prompt.prompt has a value or there are attachments.
            if os.getenv("LLM_LMSTUDIO_DEBUG") == "1":
                print("LMSTUDIO DEBUG: Building message for current turn with no text and no encodable images. Sending empty text content.", file=sys.stderr)
            msgs.append({"role": "user", "content": ""}) # Default to empty string if nothing else
        else:
            msgs.append({"role": "user", "content": current_turn_content_parts})
        
        return msgs

    # --------------------------------------------------------------------- #
    #  Execute                                                              #
    # --------------------------------------------------------------------- #
    def execute(self, prompt: llm.Prompt, stream: bool, response: llm.Response, conversation=None):
        # --- Is Execute even called? ---
        # import sys # Ensure sys is imported if not already
        # print("LMSTUDIO DEBUG: Entered LMStudioModel.execute()", file=sys.stderr)
        # --- End ---

        # --- Auto-loading Logic ---
        if not self._is_model_loaded():
            if not self._attempt_load_model():
                raise llm.ModelError(f"Failed to load model '{self.raw_id}' via 'lms load'. Please load it manually in LM Studio.")
            else:
                time.sleep(1) # Add a small delay after successful load confirmation
        # --- End Auto-loading Logic ---

        # --- Prepare Payload --- #
        # Determine the correct URL based on whether a schema is present
        if hasattr(prompt, 'schema') and prompt.schema:
            # Force /v1 endpoint for schema requests as per LM Studio docs
            url = f"{self.base}/v1/chat/completions"
            # print(f"LMSTUDIO DEBUG: Using forced /v1 endpoint for schema request: {url}", file=sys.stderr) # Add debug
        else:
            # Use the detected API path prefix for standard requests
            url = f"{self.base}{self.api_path_prefix}/chat/completions"

        messages = self._build_messages(prompt, conversation)
        payload = {"model": self.raw_id, "messages": messages}

        # Handle prompt.options (which should be an Options instance when called via model.prompt)
        if prompt.options:
            # Convert the Options object to a dictionary
            options_dict = prompt.options.model_dump(exclude_none=True)
            payload.update(options_dict)

        # Handle --schema flag from llm
        if hasattr(prompt, 'schema') and prompt.schema:
            # Use LM Studio's specific format for JSON schema output
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {  # Add the required outer object
                    "name": "llm_generated_schema", # Provide a default name
                    "schema": prompt.schema        # Nest the actual schema here
                }
            }
            # Force stream=false when schema is used, as per LM Studio examples
            stream = False # Override the input stream parameter
            # print("LMSTUDIO DEBUG: Forcing stream=False due to schema presence.", file=sys.stderr) # Add debug

        # Set stream value in payload *after* potential override
        if stream:
            payload["stream"] = True
        else:
            # Ensure stream: false is explicitly set if needed
             payload["stream"] = False

        # --- End Prepare Payload --- #

        # --- Execute API Call --- #
        # --- Debug Print --- #
        # import sys, json
        # print(f"LMSTUDIO DEBUG: Sending payload to {url}:\n{json.dumps(payload, indent=2)}", file=sys.stderr)
        # --- End Debug Print --- #

        # Determine appropriate timeout
        request_timeout = TIMEOUT # Start with default
        if hasattr(prompt, 'schema') and prompt.schema:
             # Increase timeout for potentially slower schema requests
             request_timeout = max(TIMEOUT, 30.0) # Use 30s or default, whichever is larger
             # print(f"LMSTUDIO DEBUG: Using increased timeout for schema request: {request_timeout}s", file=sys.stderr)

        try:
            # Use the determined timeout value
            r = requests.post(url, json=payload, stream=stream, timeout=request_timeout)
            r.raise_for_status()
        except requests.exceptions.Timeout:
             # Specific handling for timeout error
             raise llm.ModelError(f"LM Studio request timed out after {request_timeout} seconds. Try increasing LMSTUDIO_TIMEOUT environment variable or using a faster model.")
        except requests.RequestException as e:
             is_model_not_found = False
             try:
                 if e.response is not None:
                     err_data = e.response.json()
                     if isinstance(err_data, dict) and err_data.get("error", {}).get("code") == "model_not_found":
                         is_model_not_found = True
             except Exception:
                 pass # Ignore JSON parsing errors here

             if is_model_not_found:
                 raise llm.ModelError(f"Model '{self.raw_id}' not found by LM Studio server at {url}, even after attempting auto-load. Is it correctly specified and loadable?")
             else:
                 raise llm.ModelError(f"LM Studio request failed: {e}")
        # --- End Execute API Call --- #

        # --- Process Response --- #
        if stream:
            accumulated_content = ""
            accumulated_tool_calls = []
            current_tool_calls = [] # Stores partially built tool calls
            finish_reason = None

            for line in r.iter_lines():
                if not line or line == b"data: [DONE]" or not line.startswith(b"data:"):
                    continue
                try:
                    chunk_data = line.decode("utf-8")[5:].strip()
                    if not chunk_data:
                         continue
                    chunk = json.loads(chunk_data)
                    choice = chunk["choices"][0]
                    delta = choice.get("delta", {})
                    finish_reason = choice.get("finish_reason") # Capture finish reason from last chunk

                    # Accumulate content
                    token = delta.get("content") or ""
                    if token:
                        accumulated_content += token
                        yield token

                    # Accumulate tool calls
                    if delta.get("tool_calls"):
                        for tc_delta in delta["tool_calls"]:
                            index = tc_delta["index"]
                            # Ensure current_tool_calls list is long enough
                            while len(current_tool_calls) <= index:
                                current_tool_calls.append({
                                    "id": "", "type": "function",
                                    "function": {"name": "", "arguments": ""}
                                })
                            
                            # Update the specific tool call at the given index
                            if tc_delta.get("id"):
                                current_tool_calls[index]["id"] += tc_delta["id"]
                            if tc_delta.get("function", {}).get("name"):
                                current_tool_calls[index]["function"]["name"] += tc_delta["function"]["name"]
                            if tc_delta.get("function", {}).get("arguments"):
                                current_tool_calls[index]["function"]["arguments"] += tc_delta["function"]["arguments"]

                    if finish_reason is not None:
                        break # Stop processing stream after finish reason
                except Exception:
                    # TODO: Maybe add debug logging here?
                    continue # Ignore parsing errors, etc.
            
            # After stream finishes, process accumulated tool calls if reason indicates it
            if finish_reason == "tool_calls":
                 # Finalize accumulated tool calls (copying to avoid issues if stream is re-read)
                 accumulated_tool_calls = [tc.copy() for tc in current_tool_calls]
                 response.tool_calls(accumulated_tool_calls)

        else: # Non-streaming
            # print("LMSTUDIO DEBUG: Entered non-streaming response handling.", file=sys.stderr) # DEBUG
            try:
                raw_text = r.text # Get raw text first
                # print(f"LMSTUDIO DEBUG: Raw API Response Text:\n{raw_text}", file=sys.stderr) # DEBUG
                res = r.json()
                # print(f"LMSTUDIO DEBUG: Parsed API Response JSON:\n{json.dumps(res, indent=2)}", file=sys.stderr) # DEBUG
            except json.JSONDecodeError as e:
                 print(f"LMSTUDIO ERROR: Failed to decode JSON response: {e}", file=sys.stderr)
                 print(f"LMSTUDIO DEBUG: Failing raw text was: {raw_text}", file=sys.stderr)
                 raise llm.ModelError("Failed to decode JSON response from LM Studio.")
            except Exception as e:
                 print(f"LMSTUDIO ERROR: Unexpected error processing response: {e}", file=sys.stderr)
                 raise llm.ModelError("Unexpected error processing LM Studio response.")

            choice = res.get("choices", [{}])[0] # Safer access
            finish_reason = choice.get("finish_reason")
            # print(f"LMSTUDIO DEBUG: Finish Reason: {finish_reason}", file=sys.stderr) # DEBUG

            # Extract content
            message_content = choice.get("message", {}).get("content")
            text = message_content if message_content is not None else "" # Handle potential None
            text = text.strip() # Strip leading/trailing whitespace
            # print(f"LMSTUDIO DEBUG: Extracted content text (stripped): {repr(text)}", file=sys.stderr) # DEBUG

            # Handle tool calls (Should not happen with schema)
            if finish_reason == "tool_calls":
                # print("LMSTUDIO DEBUG: Handling tool calls (unexpected for schema).", file=sys.stderr) # DEBUG
                tool_calls_data = choice.get("message", {}).get("tool_calls")
                if tool_calls_data:
                    response.tool_calls(tool_calls_data)

            # Record usage
            usage = res.get("usage", {})
            if usage:
                # print(f"LMSTUDIO DEBUG: Setting usage: {usage}", file=sys.stderr) # DEBUG
                response.set_usage(input=usage.get("prompt_tokens", 0),
                                   output=usage.get("completion_tokens", 0))
            else:
                 # print("LMSTUDIO DEBUG: No usage info found in response.", file=sys.stderr) # DEBUG
                 pass # No need to print if no usage info

            # Yield the content instead of returning
            yield text

            # Return content
            # print(f"LMSTUDIO DEBUG: Returning stripped text list: {[text]}", file=sys.stderr) # DEBUG
            # return [text] # Original return
            # return text # Second attempt (raw string)

        # --- End Process Response --- #

# ------------------------  Async Model  ------------------------------------ #
class LMStudioAsyncModel(llm.AsyncModel):
    """Async version of the chat/completion model class."""
    can_stream: bool = True
    attachment_types = { # Task 1.1
        "image/png",
        "image/jpeg",
        "image/gif",
        "image/webp",
    }

    class Options(LMStudioModel.Options):
        # Inherit options from the sync version
        pass

    def __init__(self, model_id: str, base_url: str, raw_id: str, api_path_prefix: str, *, supports_images: bool = False, metadata: dict = None, display_suffix: str = ""):
        self.model_id = model_id # Clean ID
        self.raw_id = raw_id
        self.base = base_url
        self.api_path_prefix = api_path_prefix
        self.supports_images = supports_images
        self.metadata = metadata or {}
        self.supports_schema = True
        self.display_suffix = display_suffix # Store for __str__

    def __str__(self):
        """Return the model ID with its display suffix for listings."""
        return f"{self.model_id}{self.display_suffix}"

    def inspect(self):
        """Return model metadata for the 'llm inspect' command."""
        return self.metadata

    # Async Check/Load Helpers (using sync versions for now, could be made async)
    def _is_model_loaded(self) -> bool:
        # NOTE: For simplicity, re-using the synchronous check.
        # A fully async implementation might use httpx here.
        # This might require creating a sync LMStudioModel instance temporarily or refactoring _is_model_loaded.
        # For now, keeping it simple.
        sync_model = LMStudioModel(self.model_id, self.base, self.raw_id, self.api_path_prefix, supports_images=self.supports_images, metadata=self.metadata)
        return sync_model._is_model_loaded()

    def _attempt_load_model(self):
        # NOTE: Re-using the synchronous load attempt.
        sync_model = LMStudioModel(self.model_id, self.base, self.raw_id, self.api_path_prefix, supports_images=self.supports_images, metadata=self.metadata)
        return sync_model._attempt_load_model()

    def _encode_attachments(self, prompt: llm.Prompt) -> List[dict]:
        """Encode attachments from the prompt using llm.Attachment API.""" # Task 1.2 (mirrors sync version)
        encoded_attachments = []
        if not prompt.attachments:
            return encoded_attachments

        for attachment in prompt.attachments:
            if self.supports_images:
                try:
                    resolved_type = attachment.resolve_type()
                    if resolved_type in self.attachment_types:
                        base64_content = attachment.base64_content()
                        data_uri = f"data:{resolved_type};base64,{base64_content}"
                        encoded_attachments.append(
                            {"type": "image_url", "image_url": {"url": data_uri}}
                        )
                        if os.getenv("LLM_LMSTUDIO_DEBUG") == "1":
                            print(f"LMSTUDIO DEBUG: Encoded image attachment (async): {attachment.path or attachment.url or 'content'} as {resolved_type}.", file=sys.stderr)
                    elif os.getenv("LLM_LMSTUDIO_DEBUG") == "1":
                         print(f"LMSTUDIO DEBUG: Attachment type {resolved_type} not in model's supported image types (async). Skipping {attachment.path or attachment.url or 'content'}.", file=sys.stderr)
                except Exception as e:
                    print(f"LMSTUDIO WARN: Could not process attachment (async) {attachment.path or attachment.url or 'content'}: {e}. Skipping.", file=sys.stderr)
            elif os.getenv("LLM_LMSTUDIO_DEBUG") == "1":
                 print(f"LMSTUDIO DEBUG: Model {self.model_id} does not support images, but attachment (async) {attachment.path or attachment.url or 'content'} was provided. Ignoring.", file=sys.stderr)
        return encoded_attachments

    def _build_messages(self, prompt: llm.Prompt, conversation) -> List[dict]:
        # For async, we will call the synchronous _build_messages for now.
        # This is because _encode_attachments within it is synchronous due to llm.Attachment API.
        # If llm.Attachment provided async methods, this could be fully async.
        # Create a temporary sync model instance to call its _build_messages.
        # This is not ideal but avoids duplicating complex logic unneccessarily until llm.Attachment is async.
        sync_model_instance = LMStudioModel(
            model_id=self.model_id, 
            base_url=self.base, 
            raw_id=self.raw_id, 
            api_path_prefix=self.api_path_prefix, 
            supports_images=self.supports_images, 
            metadata=self.metadata
        )
        return sync_model_instance._build_messages(prompt, conversation)

    async def execute(
        self,
        prompt: llm.Prompt,
        stream: bool,
        response: llm.Response,
        conversation=None
    ) -> AsyncGenerator[str, None]:

        # --- Auto-loading Logic (using sync helper) ---
        if not self._is_model_loaded():
            if not self._attempt_load_model():
                raise llm.ModelError(f"Failed to load model '{self.raw_id}' via 'lms load'. Please load it manually in LM Studio.")
            else:
                # No async sleep needed here as load itself is sync
                pass
        # --- End Auto-loading Logic ---

        # --- Prepare Payload (same logic as sync) ---
        if hasattr(prompt, 'schema') and prompt.schema:
            url = f"{self.base}/v1/chat/completions"
        else:
            url = f"{self.base}{self.api_path_prefix}/chat/completions"

        messages = self._build_messages(prompt, conversation)
        payload = {"model": self.raw_id, "messages": messages}

        # Handle prompt.options (which should be an Options instance when called via model.prompt)
        if prompt.options:
            # Convert the Options object to a dictionary
            options_dict = prompt.options.model_dump(exclude_none=True)
            payload.update(options_dict)

        # Handle --schema flag from llm
        if hasattr(prompt, 'schema') and prompt.schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "llm_generated_schema",
                    "schema": prompt.schema
                }
            }
            stream = False

        if stream:
            payload["stream"] = True
        else:
            payload["stream"] = False
        # --- End Prepare Payload ---

        # Determine appropriate timeout
        request_timeout = TIMEOUT
        if hasattr(prompt, 'schema') and prompt.schema:
             request_timeout = max(TIMEOUT, 30.0)

        # --- Execute API Call (Async) ---
        try:
            async with httpx.AsyncClient(timeout=request_timeout) as client:
                if stream:
                    async with client.stream("POST", url, json=payload) as r:
                        r.raise_for_status()
                        accumulated_tool_calls = []
                        current_tool_calls = []
                        finish_reason = None

                        async for line in r.aiter_lines():
                            if not line or line == "data: [DONE]" or not line.startswith("data:"):
                                continue
                            try:
                                chunk_data = line[5:].strip()
                                if not chunk_data:
                                    continue
                                chunk = json.loads(chunk_data)
                                choice = chunk["choices"][0]
                                delta = choice.get("delta", {})
                                finish_reason = choice.get("finish_reason")

                                # Accumulate content
                                token = delta.get("content") or ""
                                if token:
                                    yield token

                                # Accumulate tool calls (logic adapted from sync)
                                if delta.get("tool_calls"):
                                    for tc_delta in delta["tool_calls"]:
                                        index = tc_delta["index"]
                                        while len(current_tool_calls) <= index:
                                            current_tool_calls.append({
                                                "id": "", "type": "function",
                                                "function": {"name": "", "arguments": ""}
                                            })
                                        if tc_delta.get("id"):
                                            current_tool_calls[index]["id"] += tc_delta["id"]
                                        if tc_delta.get("function", {}).get("name"):
                                            current_tool_calls[index]["function"]["name"] += tc_delta["function"]["name"]
                                        if tc_delta.get("function", {}).get("arguments"):
                                            current_tool_calls[index]["function"]["arguments"] += tc_delta["function"]["arguments"]

                                if finish_reason is not None:
                                    break
                            except Exception:
                                continue # Ignore parsing errors

                        # After stream finishes, process tool calls
                        if finish_reason == "tool_calls":
                             accumulated_tool_calls = [tc.copy() for tc in current_tool_calls]
                             response.tool_calls(accumulated_tool_calls)
                        
                        # Potential place to check for usage in final stream chunk if API supports it

                else: # Non-streaming async
                    r = await client.post(url, json=payload)
                    r.raise_for_status()
                    try:
                        raw_text = r.text
                        res = r.json()
                    except json.JSONDecodeError as e:
                         print(f"LMSTUDIO ERROR: Failed to decode JSON response: {e}", file=sys.stderr)
                         print(f"LMSTUDIO DEBUG: Failing raw text was: {raw_text}", file=sys.stderr)
                         raise llm.ModelError("Failed to decode JSON response from LM Studio.")
                    except Exception as e:
                         print(f"LMSTUDIO ERROR: Unexpected error processing response: {e}", file=sys.stderr)
                         raise llm.ModelError("Unexpected error processing LM Studio response.")

                    choice = res.get("choices", [{}])[0]
                    finish_reason = choice.get("finish_reason")

                    message_content = choice.get("message", {}).get("content")
                    text = message_content if message_content is not None else ""
                    text = text.strip()

                    if finish_reason == "tool_calls":
                        tool_calls_data = choice.get("message", {}).get("tool_calls")
                        if tool_calls_data:
                            response.tool_calls(tool_calls_data)

                    usage = res.get("usage", {})
                    if usage:
                        response.set_usage(input=usage.get("prompt_tokens", 0),
                                           output=usage.get("completion_tokens", 0))

                    yield text # Yield the single result

        except httpx.TimeoutException:
            raise llm.ModelError(f"LM Studio request timed out after {request_timeout} seconds (async). Try increasing LMSTUDIO_TIMEOUT.")
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

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        try:
            r = requests.post(
                f"{self.base}{self.api_path_prefix}/embeddings",
                json={"model": self.raw_id, "input": texts},
                timeout=TIMEOUT,
            )
            r.raise_for_status()
            data = r.json()
            return [item["embedding"] for item in data["data"]]
        except requests.RequestException as e:
            raise llm.ModelError(f"LM Studio embeddings request failed: {e}")
        except Exception as e:
            raise llm.ModelError(f"Unexpected embeddings response: {e}")