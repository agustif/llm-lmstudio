#!/usr/bin/env python3
"""
End-to-end synchronous LM Studio plugin test recorded with pytest-vcr.

This verifies that the plugin registers models with llm, can create prompts,
issues HTTP requests (captured by VCR), and parses responses/usage metadata.
"""

import logging
import os

import llm
import pytest

import llm_lmstudio
from llm_lmstudio import LMStudioAsyncModel, LMStudioModel

logging.basicConfig()
vcr_logger = logging.getLogger("vcr")
vcr_logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
vcr_logger.addHandler(handler)


@pytest.fixture(autouse=True)
def reset_lmstudio_plugin_caches():
    llm_lmstudio._cache.clear()
    llm_lmstudio._errors.clear()
    yield
    llm_lmstudio._cache.clear()
    llm_lmstudio._errors.clear()


@pytest.fixture(scope="module")
def vcr_cassette_dir(request):
    return os.path.join(os.path.dirname(__file__), "cassettes")


@pytest.mark.vcr(record_mode="once")
def test_lmstudio_sync_prompt_round_trip():
    model_id = os.getenv("LMSTUDIO_TEST_MODEL_ID", "lmstudio/qwen3-4b")
    prompt_text = "Give me a cheerful greeting in one sentence."

    model = llm.get_model(model_id)
    assert model is not None, f"Model '{model_id}' was not registered"
    assert isinstance(model, LMStudioModel), (
        "Retrieved model is not provided by the LM Studio plugin"
    )

    response = model.prompt(prompt_text, stream=False)
    assert response is not None, "plugin prompt() should return a response object"

    text = response.text()
    assert "Hi there! Have a wonderful day!" in text
    assert isinstance(text, str), "response.text() should return a string"
    assert text.strip(), "Model returned empty response text"

    usage = response.usage()
    assert hasattr(usage, "input"), "Usage metadata missing 'input'"
    assert hasattr(usage, "output"), "Usage metadata missing 'output'"

    # Ensure the metadata looks reasonable (non-negative token counts, if provided)
    if usage.input is not None:
        assert usage.input >= 0
    if usage.output is not None:
        assert usage.output >= 0


@pytest.mark.asyncio
@pytest.mark.vcr(record_mode="once")
async def test_lmstudio_async_prompt_round_trip():
    model_id = os.getenv("LMSTUDIO_TEST_MODEL_ID", "lmstudio/qwen3-4b")
    prompt_text = "Give me a cheerful greeting in one sentence."

    model = llm.get_async_model(model_id)
    assert model is not None, f"Model '{model_id}' was not registered"
    assert isinstance(model, LMStudioAsyncModel), (
        "Retrieved async model is not provided by the LM Studio plugin"
    )

    response = await model.prompt(prompt_text, stream=False)
    assert response is not None, "plugin prompt() should return a response object"

    text = await response.text()
    assert "Have a day as bright and cheerful as the morning sun!" in text
    assert isinstance(text, str), "response.text() should return a string"
    assert text.strip(), "Model returned empty response text"

    usage = await response.usage()
    assert hasattr(usage, "input"), "Usage metadata missing 'input'"
    assert hasattr(usage, "output"), "Usage metadata missing 'output'"

    # Ensure the metadata looks reasonable (non-negative token counts, if provided)
    if usage.input is not None:
        assert usage.input >= 0
    if usage.output is not None:
        assert usage.output >= 0
