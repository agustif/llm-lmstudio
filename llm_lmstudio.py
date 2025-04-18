import json
import requests  # HTTP library for calling the local LMStudio server
import llm
import os  # Import the os module
from pydantic import Field
from typing import Optional, List, Dict, Any

# Base URL of the LMStudio API. Reads from LMSTUDIO_API_BASE environment variable,
# falling back to the default localhost:1234 if not set.
LMSTUDIO_API_BASE = os.getenv("LMSTUDIO_API_BASE", "http://localhost:1234")

# --- Helper Function for fetching models --- 
_fetched_models: Optional[List[Dict[str, Any]]] = None
_fetch_error: Optional[Exception] = None

def _get_lmstudio_models() -> tuple[Optional[List[Dict[str, Any]]], Optional[Exception]]:
    """Fetches model list from LMStudio API, caching the result for the lifetime of the process."""
    global _fetched_models, _fetch_error
    if _fetched_models is not None or _fetch_error is not None:
        return _fetched_models, _fetch_error
    try:
        resp = requests.get(f"{LMSTUDIO_API_BASE}/v1/models")
        resp.raise_for_status()
        data = resp.json()
        _fetched_models = data.get("data", [])
        return _fetched_models, None
    except requests.RequestException as e:
        _fetch_error = e
        return None, e
    except json.JSONDecodeError as e:
        _fetch_error = e # Treat JSON errors like request errors
        return None, e

# --- Registration Hooks --- 

@llm.hookimpl
def register_models(register):
    """Discover and register LMStudio *chat/completion* models."""
    models, error = _get_lmstudio_models()
    if error:
        # Raise the error here, as chat models are primary
        raise llm.ModelError(f"Could not connect to LMStudio server at {LMSTUDIO_API_BASE} - {error}")
    if not models:
        return
    
    for m in models:
        model_id = m.get("id")
        if not model_id:
            continue
        # Heuristic: Skip embedding models in this hook. 
        # Relies on model ID containing "embed".
        is_embedding = "embed" in model_id.lower()
        if is_embedding:
            continue 
        
        # Register chat/completion models
        model_instance = LMStudio(model_id)
        alias = None
        if "/" in model_id:
            alias = model_id.split("/")[-1]
            alias = alias.rsplit(".", 1)[0]
        register(model_instance, aliases=([alias] if alias else (())))

@llm.hookimpl
def register_embedding_models(register):
    """Discover and register LMStudio *embedding* models."""
    models, error = _get_lmstudio_models()
    if error:
        # Don't raise here, just warn/log, as embedding models are secondary
        print(f"Warning: Could not retrieve LMStudio models to find embedding models - {error}")
        return
    if not models:
        return
        
    for m in models:
        model_id = m.get("id")
        if not model_id:
            continue
        # Heuristic: Only register embedding models in this hook.
        # Relies on model ID containing "embed".
        is_embedding = "embed" in model_id.lower()
        if is_embedding:
            register(LMStudioEmbeddingModel(model_id))
            # No aliases needed for embedding models usually

class LMStudio(llm.Model):
    """
    Model class for LMStudio text-generation models. 
    Handles both one-shot completions and multi-turn chat using the LMStudio HTTP API.
    """
    def __init__(self, model_id: str):
        # Unique identifier for this model (used with -m option in CLI)
        self.model_id = f"lmstudio/{model_id}"  # Prefix with "lmstudio/" to group under LMStudio
        # Alternatively, you could use model_id as-is. Prefixed ID helps identify the provider.

    # Allow streaming output 
    can_stream: bool = True

    # Define supported generation options (mirroring OpenAI parameters)
    class Options(llm.Options):
        temperature: Optional[float] = Field(None, description="Sampling temperature")
        top_p: Optional[float]      = Field(None, description="Nucleus sampling probability")
        max_tokens: Optional[int]   = Field(None, description="Maximum tokens to generate (use -1 for no limit)")
        stop: Optional[List[str]]   = Field(None, description="Stop sequences where generation halts")
        # (Add other OpenAI-style parameters as needed, e.g. presence_penalty, frequency_penalty)

    def _build_messages(self, prompt: llm.Prompt, conversation) -> List[dict]:
        """Construct the OpenAI-formatted messages array for the request."""
        messages = []
        # Include system prompt if provided
        if prompt.system:
            messages.append({"role": "system", "content": prompt.system})
        # If there is a conversation with prior turns, include them
        if conversation:
            # conversation.responses holds previous turns 
            # Each response has a prompt (which contains the user message and system) and possibly text.
            for prev in conversation.responses:
                # If the previous prompt had a system message and it's different from last used, include it
                if prev.prompt.system:
                    messages.append({"role": "system", "content": prev.prompt.system})
                # User's previous prompt
                messages.append({"role": "user", "content": prev.prompt.prompt})
                # Assistant's previous response
                messages.append({"role": "assistant", "content": prev.text_or_raise()})
        # Add the current user prompt last
        messages.append({"role": "user", "content": prompt.prompt})
        return messages

    def execute(self, prompt: llm.Prompt, stream: bool, response: llm.Response, conversation=None):
        """Execute a prompt or chat conversation against the LMStudio API."""
        # Build request payload
        if conversation or prompt.system:
            # Use the chat/completions endpoint with messages
            url = f"{LMSTUDIO_API_BASE}/v1/chat/completions"
            payload = {"model": self.model_id.split("lmstudio/", 1)[-1],  # LMStudio needs the raw model ID
                       "messages": self._build_messages(prompt, conversation)}
        else:
            # No conversation context and no system prompt: we can use completion endpoint
            url = f"{LMSTUDIO_API_BASE}/v1/completions"
            payload = {"model": self.model_id.split("lmstudio/", 1)[-1], # LMStudio needs the raw model ID
                       "prompt": prompt.prompt}
        # Include generation options from prompt.options
        opts = prompt.options.model_dump(exclude_none=True)
        # Remove any fields not part of OpenAI payload:
        opts.pop("model", None)  # model is already set separately
        payload.update(opts)
        # If streaming, request streaming from API if supported (LMStudio supports "stream": true)
        if stream:
            payload["stream"] = True

        # Make the HTTP request to LMStudio
        try:
            api_response = requests.post(url, json=payload, stream=stream)
            api_response.raise_for_status()
        except requests.RequestException as e:
            raise llm.ModelError(f"LMStudio request failed: {e}")

        # Handle the response
        if stream:
            # LMStudio streams responses as events. Here we read chunks as they arrive.
            for line in api_response.iter_lines():
                if line:
                    # Each line may contain a JSON partial message or data.
                    # LMStudio (OpenAI style) streaming sends lines starting with "data: {...}" per chunk.
                    if line.strip() == b'data: [DONE]':
                        break
                    if not line.startswith(b'data:'):
                        continue
                    try:
                        line_data = line.decode("utf-8").lstrip("data: ").strip()
                        if not line_data:
                            continue
                        decoded = json.loads(line_data)
                    except json.JSONDecodeError as e:
                        # Handle potential errors in decoding or incomplete JSON chunks
                        # print(f"JSON decoding error: {e} on line: {line}") # Optional: for debugging
                        continue
                    except Exception as e:
                        # print(f"Other error during stream processing: {e}") # Optional: for debugging
                        continue
                    
                    # Extract content based on endpoint type (chat vs completion)
                    chunk = ""
                    if url.endswith("/v1/chat/completions"):
                        delta = decoded.get("choices",[{}])[0].get("delta", {})
                        if "content" in delta:
                            chunk = delta["content"] or "" # Handle potential null content
                    elif url.endswith("/v1/completions"):
                         text = decoded.get("choices",[{}])[0].get("text", "")
                         if text:
                            chunk = text
                    
                    if chunk:
                       yield chunk
                    
                    # Check finish reason (applies to both endpoints in streaming)
                    if decoded.get("choices",[{}])[0].get("finish_reason") is not None:
                        break
        else:
            # Non-streaming: return the full completion text
            result = api_response.json()
            # Extract the text content from the response (OpenAI format)
            try:
                choice = result["choices"][0]
                # Handle both chat completion and text completion response structures
                if "message" in choice and "content" in choice["message"]:
                    text = choice["message"]["content"]
                elif "text" in choice:
                    text = choice["text"]
                else:
                    text = ""
            except (KeyError, IndexError, TypeError):
                raise llm.ModelError(f"Unexpected response format from LMStudio: {result}")
            
            # Optionally, record token usage if provided
            usage = result.get("usage")
            if usage and isinstance(usage, dict):
                response.set_usage(input=usage.get("prompt_tokens", 0), output=usage.get("completion_tokens", 0))
            
            return [text]  # return as a list containing the complete response text

# (Optional) If we wanted to support embedding models, we would define a similar class:
class LMStudioEmbeddingModel(llm.EmbeddingModel):
    def __init__(self, model_id: str):
        self.model_id = f"lmstudio/{model_id}"

    # Renamed from execute to embed_batch to match llm >= 0.13
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Call LMStudio embedding endpoint
        # Note: LMStudio API expects a single string or list of strings for "input"
        # LLM passes a list of texts. We'll pass them directly.
        raw_model_id = self.model_id.split("lmstudio/", 1)[-1]
        try:
            resp = requests.post(f"{LMSTUDIO_API_BASE}/v1/embeddings",
                                 json={"model": raw_model_id, "input": texts})
            resp.raise_for_status()
            data = resp.json()
            # The response format is a list of embedding objects, one per input text
            embeddings = [item["embedding"] for item in data["data"]]
            # Return list of embedding vectors (list of lists of floats)
            return embeddings
        except requests.RequestException as e:
            raise llm.ModelError(f"LMStudio embedding request failed: {e}")
        except (KeyError, IndexError, TypeError) as e:
             raise llm.ModelError(f"Unexpected response format from LMStudio embedding endpoint: {data} - Error: {e}") 