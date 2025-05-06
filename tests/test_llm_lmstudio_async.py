# tests/test_llm_lmstudio_async.py
import pytest
import llm
# import llm_lmstudio # Import no longer needed if relying on entry points
import json
# import respx # No longer using respx
import httpx
# from unittest.mock import patch # No longer needed

# Mark all tests in this file as asyncio
pytestmark = pytest.mark.asyncio

# --- Test Data ---

# Target model ID for VCR tests should be the plain ID llm uses to find the model.
# The plugin internally maps this to the raw_id for API calls.
# This assumes 'llava-v1.5-7b' is the ID as recognized by llm after plugin registration.
MODEL_ID = "llava-v1.5-7b" 
BASE_URL = "http://localhost:1234" # VCR will handle this

# --- Tests ---

@pytest.mark.vcr
async def test_get_async_model():
    """Test retrieving the specific async model instance."""
    # This test focuses on ensuring discovery works via VCR for the specific model
    model = llm.get_async_model(MODEL_ID)
    assert isinstance(model, llm.AsyncModel)
    assert model.model_id == MODEL_ID
    from llm_lmstudio import LMStudioAsyncModel
    assert isinstance(model, LMStudioAsyncModel)


@pytest.mark.vcr
async def test_async_prompt_non_streaming():
    """Test a basic non-streaming async prompt using model.response()."""
    # NOTE: Requires cassette generated against a live LM Studio server
    # with MODEL_ID loaded. Assertions MUST be updated after recording.
    model = llm.get_async_model(MODEL_ID)

    prompt_text = "Say hello"
    response = await model.prompt(prompt_text, stream=False)

    # !!! IMPORTANT: Update this assertion after first VCR run !!!
    response_text = await response.text() # Await the text
    assert response_text is not None and len(response_text) > 0 # Generic check
    assert response.model.model_id == MODEL_ID
    # Check usage - these might also need adjustment based on the actual model
    # response.usage is a coroutine and needs to be awaited
    usage = await response.usage()

    assert hasattr(usage, "input"), "Usage object is missing 'input'"
    assert hasattr(usage, "output"), "Usage object is missing 'output'"
    assert usage.input > 0
    assert usage.output > 0

    print(response)
    # print(await response.usage()) # Removed to prevent RuntimeWarning, usage is already tested


@pytest.mark.vcr
async def test_async_prompt_streaming():
    """Test a basic streaming async prompt using model.response()."""
    # NOTE: Requires cassette generated against a live LM Studio server
    # with MODEL_ID loaded. Assertions MUST be updated after recording.
    model = llm.get_async_model(MODEL_ID)

    prompt_text = "Say hello streaming"
    response_stream = await model.prompt(prompt_text, stream=True)

    chunks = [chunk async for chunk in response_stream]
    full_response = "".join(chunks)

    # !!! IMPORTANT: Update this assertion after first VCR run !!!
    assert full_response is not None and len(full_response) > 0 # Generic check


@pytest.mark.vcr
async def test_async_prompt_schema():
    """Test non-streaming async prompt with a schema using model.response()."""
    # NOTE: Requires cassette generated against a live LM Studio server
    # with MODEL_ID loaded. Assertions MUST be updated after recording.
    model = llm.get_async_model(MODEL_ID)

    prompt = "Invent a test dog"
    schema_def = 'name, age int, one_sentence_bio'
    schema_dict = llm.schema_dsl(schema_def)

    response = await model.prompt(prompt, stream=False, schema=schema_dict)
    response_text = await response.text() # Await the text

    # !!! IMPORTANT: Update assertions after first VCR run !!!
    try:
        data = json.loads(response_text)
        assert "name" in data
        assert "age" in data
        assert "one_sentence_bio" in data
        assert isinstance(data["age"], int)
    except json.JSONDecodeError:
        pytest.fail(f"Response was not valid JSON: {response_text}")

    assert response.model.model_id == MODEL_ID
    # Check usage - response.usage is a coroutine
    usage = await response.usage()

    assert hasattr(usage, "input"), "Usage object is missing 'input'"
    assert hasattr(usage, "output"), "Usage object is missing 'output'"
    assert usage.input > 0
    assert usage.output > 0