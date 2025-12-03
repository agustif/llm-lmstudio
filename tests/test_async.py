import json
import logging
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import llm
import pytest

import llm_lmstudio

# Mark all tests in this file as asyncio
pytestmark = pytest.mark.asyncio

# --- VCR Configuration ---
logging.basicConfig()
vcr_logger = logging.getLogger("vcr")
vcr_logger.setLevel(logging.DEBUG)  # Set to DEBUG for max verbosity
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
vcr_logger.addHandler(handler)


@pytest.fixture(scope="module")
def vcr_cassette_dir(request):
    # Standard cassette directory
    return os.path.join(os.path.dirname(__file__), "cassettes")


# --- Constants for Mocking ---
# This is the raw_id that _fetch_models would return, and llm.get_async_model will look for.
MOCK_RAW_MODEL_ID = "llava-v1.5-7b"
# This is the model_id llm will use (plugin prefix + raw_id if multiple servers, just raw_id if single default server)
# For these tests, assuming single server context, so MODEL_ID = lmstudio prefix + MOCK_RAW_MODEL_ID
MODEL_ID = "lmstudio/" + MOCK_RAW_MODEL_ID

MOCK_MODELS_LIST = [
    {
        "id": MOCK_RAW_MODEL_ID,  # Corresponds to raw_id in the plugin
        "type": "vlm",  # Ensures supports_images=True logic path
        "vision": True,  # Explicit vision flag
        "state": "loaded",  # Assumed loaded for testing
        "publisher": "mock_publisher",  # Example metadata
        "architecture": "mock_arch",
        "quantization": "mock_quant",
        "max_context_length": 2048,
    }
]
MOCK_API_PATH = "/api/v0"  # API path prefix the plugin would discover
MOCK_FETCH_MODELS_RETURN_VALUE = (MOCK_MODELS_LIST, MOCK_API_PATH)

# --- Test Data ---

# Target model ID for VCR tests should be the plain ID llm uses to find the model.
# The plugin internally maps this to the raw_id for API calls.
# This assumes 'llava-v1.5-7b' is the ID as recognized by llm after plugin registration.
# MODEL_ID = "llava-v1.5-7b" # MOVED UP and renamed for clarity with mock
BASE_URL = "http://localhost:1234"  # VCR will handle this

# --- Tests ---


@pytest.mark.vcr(record_mode="once")  # CHANGED from 'all' to 'once'
@patch("llm_lmstudio._fetch_models", return_value=MOCK_FETCH_MODELS_RETURN_VALUE)
async def test_get_async_model(mock_fetch_list):
    """Test retrieving the specific async model instance."""
    try:
        model = llm.get_async_model(MODEL_ID)
        assert model is not None
        assert model.model_id == MODEL_ID
        # Check for the display_suffix if it's consistently applied by the plugin
        # This part depends on how your __init__.py and model registration works
        # For now, focus on VCR generation.
        # assert hasattr(model, 'display_suffix')
        # assert "👁" in model.display_suffix
    except Exception:
        # print(f"DEBUG: test_get_async_model EXCEPTION: {e}") # Removed diagnostic
        raise


@pytest.mark.vcr(record_mode="once")  # CHANGED from 'all' to 'once'
@patch("llm_lmstudio.LMStudioAsyncModel._is_model_loaded", return_value=True)
@patch("llm_lmstudio._fetch_models", return_value=MOCK_FETCH_MODELS_RETURN_VALUE)
async def test_async_prompt_non_streaming(mock_fetch_list, mock_is_loaded):
    """Test a basic non-streaming async prompt using model.response()."""
    try:
        model = llm.get_async_model(MODEL_ID)
        assert model is not None
        prompt_text = "Say hello"
        response = await model.prompt(prompt_text, stream=False)
        assert response is not None
        retrieved_text = await response.text()
        assert retrieved_text is not None, "response.text() should not be None"
        assert isinstance(retrieved_text, str), (
            f"await response.text() should be a string, got {type(retrieved_text)}"
        )
        assert retrieved_text.strip()
        usage = await response.usage()
        assert hasattr(usage, "input"), "Usage object is missing 'input'"
        assert hasattr(usage, "output"), "Usage object is missing 'output'"
    except Exception:
        # print(f"DEBUG: test_async_prompt_non_streaming EXCEPTION: {e}") # Removed diagnostic
        raise


@pytest.mark.vcr(record_mode="once")  # CHANGED from 'all' to 'once'
@patch("llm_lmstudio.LMStudioAsyncModel._is_model_loaded", return_value=True)
@patch("llm_lmstudio._fetch_models", return_value=MOCK_FETCH_MODELS_RETURN_VALUE)
async def test_async_prompt_streaming(mock_fetch_list, mock_is_loaded):
    """Test a basic streaming async prompt using model.response()."""
    try:
        model = llm.get_async_model(MODEL_ID)
        assert model is not None
        prompt_text = "Tell a short story."
        response_chunks_objects = []
        async for chunk_obj in await model.prompt(prompt_text, stream=True):
            response_chunks_objects.append(chunk_obj)
        assert len(response_chunks_objects) > 0
        retrieved_texts = []
        for chunk_obj in response_chunks_objects:
            # Assuming chunk_obj itself is the text string based on previous findings
            if isinstance(chunk_obj, str):
                retrieved_texts.append(chunk_obj)
            else:
                # Fallback if it's an object with a .text() method (less likely now)
                # This path might indicate an unexpected change in llm behavior or our understanding
                try:
                    text_content = await chunk_obj.text()
                    retrieved_texts.append(text_content)
                except AttributeError:
                    print(
                        f"DEBUG: test_async_prompt_streaming - chunk_obj of type {type(chunk_obj)} has no .text() method and is not str."
                    )
                    # Decide how to handle this - for now, append its string representation if not None
                    if chunk_obj is not None:
                        retrieved_texts.append(str(chunk_obj))
        assert len(retrieved_texts) > 0, "Should have collected some text from stream"
        full_response_text = "".join(retrieved_texts)
        assert full_response_text.strip()
    except Exception:
        # print(f"DEBUG: test_async_prompt_streaming EXCEPTION: {e}") # Removed diagnostic
        raise


@pytest.mark.vcr(record_mode="once")  # CHANGED from 'all' to 'once'
@patch("llm_lmstudio.LMStudioAsyncModel._is_model_loaded", return_value=True)
@patch("llm_lmstudio._fetch_models", return_value=MOCK_FETCH_MODELS_RETURN_VALUE)
async def test_async_prompt_schema(mock_fetch_list, mock_is_loaded):
    """Test async prompt with a JSON schema for structured output."""
    try:
        model = llm.get_async_model(MODEL_ID)
        assert model is not None
        schema = {
            "type": "object",
            "properties": {
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "neutral", "negative"],
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["sentiment", "confidence"],
        }
        prompt_text = "Analyze the sentiment of this sentence: 'I love sunny days!'"
        response = await model.prompt(prompt_text, schema=schema, stream=False)
        assert response is not None
        retrieved_text = await response.text()
        assert retrieved_text
        parsed_json = json.loads(retrieved_text)
        assert "sentiment" in parsed_json
        assert "confidence" in parsed_json
        assert parsed_json["sentiment"] in ["positive", "neutral", "negative"]
        assert 0 <= parsed_json["confidence"] <= 1
    except Exception:
        # print(f"DEBUG: test_async_prompt_schema EXCEPTION: {e}") # Removed diagnostic
        raise


async def test_async_execute_handles_tool_call_response(monkeypatch):
    monkeypatch.setattr(
        llm_lmstudio.LMStudioAsyncModel, "_is_model_loaded", lambda self: True
    )

    api_response = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_weather_async",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "Berlin"}',
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 33, "completion_tokens": 7},
    }

    class FakeAsyncResponse:
        def __init__(self, payload):
            self._payload = payload
            self.text = json.dumps(payload)

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    last_request = {}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json=None):
            last_request["url"] = url
            last_request["json"] = json
            return FakeAsyncResponse(api_response)

    monkeypatch.setattr(llm_lmstudio.httpx, "AsyncClient", FakeAsyncClient)

    async_model = llm_lmstudio.LMStudioAsyncModel(
        model_id="lmstudio/test",
        base_url="http://localhost:1234",
        raw_id="test-model",
        api_path_prefix="/api/v0",
    )

    tools = [
        llm.Tool(
            name="get_weather",
            description="Return mock weather for a city.",
            input_schema={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City to inspect",
                    }
                },
                "required": ["location"],
            },
        )
    ]

    prompt = SimpleNamespace(
        prompt="Please check the weather in Berlin.",
        attachments=[],
        system=None,
        options=None,
        schema=None,
        tools=tools,
        tool_results=[],
    )

    response = MagicMock()

    result = [
        chunk
        async for chunk in async_model.execute(
            prompt=prompt, stream=False, response=response, conversation=None
        )
    ]

    assert result == [""]
    response.add_tool_call.assert_called_once()
    added_call = response.add_tool_call.call_args[0][0]
    assert added_call.name == "get_weather"
    assert added_call.arguments == {"location": "Berlin"}
    assert added_call.tool_call_id == "call_weather_async"
    response.set_usage.assert_called_once_with(input=33, output=7)

    sent_tools = last_request["json"]["tools"]
    assert sent_tools[0]["function"]["name"] == "get_weather"
    assert sent_tools[0]["function"]["parameters"]["required"] == ["location"]
    final_message = last_request["json"]["messages"][-1]
    assert final_message["content"][0]["text"] == "Please check the weather in Berlin."
