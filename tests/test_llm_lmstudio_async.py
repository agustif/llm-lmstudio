# tests/test_llm_lmstudio_async.py
import pytest
import llm
# import llm_lmstudio # Import no longer needed if relying on entry points
import json
# import respx # No longer using respx
import httpx
from unittest.mock import patch # ADDED
from typing import Dict, Any # Reverted typing, before_record_log removed for now
import logging # ADDED for VCR logging
import os # For cassette_library_dir

# Mark all tests in this file as asyncio
pytestmark = pytest.mark.asyncio

# --- VCR Configuration ---
logging.basicConfig()
vcr_logger = logging.getLogger("vcr")
vcr_logger.setLevel(logging.DEBUG) # Set to DEBUG for max verbosity
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s:%(message)s'))
vcr_logger.addHandler(handler)

@pytest.fixture(scope='module')
def vcr_cassette_dir(request):
    # Standard cassette directory
    return os.path.join(os.path.dirname(__file__), "cassettes")

# --- Constants for Mocking ---
# This is the raw_id that _fetch_models would return, and llm.get_async_model will look for.
MOCK_RAW_MODEL_ID = "llava-v1.5-7b"
# This is the model_id llm will use (plugin prefix + raw_id if multiple servers, just raw_id if single default server)
# For these tests, assuming single server context, so MODEL_ID = MOCK_RAW_MODEL_ID
MODEL_ID = MOCK_RAW_MODEL_ID

MOCK_MODELS_LIST = [{
    'id': MOCK_RAW_MODEL_ID, # Corresponds to raw_id in the plugin
    'type': 'vlm',          # Ensures supports_images=True logic path
    'vision': True,         # Explicit vision flag
    'state': 'loaded',      # Assumed loaded for testing
    'publisher': 'mock_publisher', # Example metadata
    'architecture': 'mock_arch',
    'quantization': 'mock_quant',
    'max_context_length': 2048
}]
MOCK_API_PATH = "/api/v0" # API path prefix the plugin would discover
MOCK_FETCH_MODELS_RETURN_VALUE = (MOCK_MODELS_LIST, MOCK_API_PATH)

# --- Test Data ---

# Target model ID for VCR tests should be the plain ID llm uses to find the model.
# The plugin internally maps this to the raw_id for API calls.
# This assumes 'llava-v1.5-7b' is the ID as recognized by llm after plugin registration.
# MODEL_ID = "llava-v1.5-7b" # MOVED UP and renamed for clarity with mock
BASE_URL = "http://localhost:1234" # VCR will handle this

# --- Tests ---

@pytest.mark.vcr(record_mode='once') # CHANGED from 'all' to 'once'
@patch('llm_lmstudio._fetch_models', return_value=MOCK_FETCH_MODELS_RETURN_VALUE)
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
    except Exception as e:
        # print(f"DEBUG: test_get_async_model EXCEPTION: {e}") # Removed diagnostic
        raise

@pytest.mark.vcr(record_mode='once') # CHANGED from 'all' to 'once'
@patch('llm_lmstudio.LMStudioAsyncModel._is_model_loaded', return_value=True)
@patch('llm_lmstudio._fetch_models', return_value=MOCK_FETCH_MODELS_RETURN_VALUE)
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
        assert isinstance(retrieved_text, str), f"await response.text() should be a string, got {type(retrieved_text)}"
        assert retrieved_text.strip()
        usage = await response.usage()
        assert hasattr(usage, "input"), "Usage object is missing 'input'"
        assert hasattr(usage, "output"), "Usage object is missing 'output'"
    except Exception as e:
        # print(f"DEBUG: test_async_prompt_non_streaming EXCEPTION: {e}") # Removed diagnostic
        raise

@pytest.mark.vcr(record_mode='once') # CHANGED from 'all' to 'once'
@patch('llm_lmstudio.LMStudioAsyncModel._is_model_loaded', return_value=True)
@patch('llm_lmstudio._fetch_models', return_value=MOCK_FETCH_MODELS_RETURN_VALUE)
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
                    print(f"DEBUG: test_async_prompt_streaming - chunk_obj of type {type(chunk_obj)} has no .text() method and is not str.")
                    # Decide how to handle this - for now, append its string representation if not None
                    if chunk_obj is not None:
                         retrieved_texts.append(str(chunk_obj))
        assert len(retrieved_texts) > 0, "Should have collected some text from stream"
        full_response_text = "".join(retrieved_texts)
        assert full_response_text.strip()
    except Exception as e:
        # print(f"DEBUG: test_async_prompt_streaming EXCEPTION: {e}") # Removed diagnostic
        raise

@pytest.mark.vcr(record_mode='once') # CHANGED from 'all' to 'once'
@patch('llm_lmstudio.LMStudioAsyncModel._is_model_loaded', return_value=True)
@patch('llm_lmstudio._fetch_models', return_value=MOCK_FETCH_MODELS_RETURN_VALUE)
async def test_async_prompt_schema(mock_fetch_list, mock_is_loaded):
    """Test async prompt with a JSON schema for structured output."""
    try:
        model = llm.get_async_model(MODEL_ID)
        assert model is not None
        schema = {
            "type": "object",
            "properties": {
                "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1}
            },
            "required": ["sentiment", "confidence"]
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
    except Exception as e:
        # print(f"DEBUG: test_async_prompt_schema EXCEPTION: {e}") # Removed diagnostic
        raise