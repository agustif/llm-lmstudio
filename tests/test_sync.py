import io
import json
from types import SimpleNamespace
from typing import List, Optional
from unittest.mock import MagicMock, patch

import llm
import pytest

import llm_lmstudio
from llm_lmstudio import LMStudioModel

# --- Fixtures ---


@pytest.fixture
def mock_model_instance_factory():
    """Factory to create LMStudioModel instances with specific image support."""

    def _factory(
        supports_images: bool,
        model_id: str = "test-vlm-model",
        base_url: str = "http://localhost:1234",
        raw_id: str = "test-vlm-raw",
        api_path: str = "/v1",
    ):
        display_suffix = ""
        if supports_images:
            display_suffix += " 👁️"
        display_suffix += " ⚒️"
        display_suffix = display_suffix.lstrip()
        if display_suffix:
            display_suffix = " " + display_suffix

        model = LMStudioModel(
            model_id=model_id,
            base_url=base_url,
            raw_id=raw_id,
            api_path_prefix=api_path,
            supports_images=supports_images,
            metadata={
                "vision": supports_images,
                "raw_lmstudio_type": "vlm" if supports_images else "llm",
            },
            display_suffix=display_suffix,
        )
        return model

    return _factory


@pytest.fixture
def vlm_model(mock_model_instance_factory):
    """An LMStudioModel instance that supports images."""
    return mock_model_instance_factory(
        supports_images=True, model_id="test-vlm", raw_id="test-vlm-raw-id"
    )


@pytest.fixture
def non_vlm_model(mock_model_instance_factory):
    """An LMStudioModel instance that does NOT support images."""
    return mock_model_instance_factory(
        supports_images=False, model_id="test-non-vlm", raw_id="test-non-vlm-raw-id"
    )


@pytest.fixture
def mock_attachment_factory(mocker):
    """Factory to create MagicMock llm.Attachment objects."""

    def _factory(
        mime_type: Optional[str] = "image/png",
        base64_content: Optional[str] = "dGVzdA==",  # "test"
        path: Optional[str] = "test.png",
        url: Optional[str] = None,
        resolve_type_raises: Optional[Exception] = None,
        base64_content_raises: Optional[Exception] = None,
    ):
        attachment = MagicMock(spec=llm.Attachment)
        attachment.path = path
        attachment.url = url

        if resolve_type_raises:
            attachment.resolve_type = mocker.Mock(side_effect=resolve_type_raises)
        else:
            attachment.resolve_type = mocker.Mock(return_value=mime_type)

        if base64_content_raises:
            attachment.base64_content = mocker.Mock(side_effect=base64_content_raises)
        else:
            attachment.base64_content = mocker.Mock(return_value=base64_content)

        return attachment

    return _factory


@pytest.fixture
def mock_prompt_factory(mocker):
    """Factory to create MagicMock llm.Prompt objects."""

    def _factory(
        prompt_text: Optional[str],
        attachments: Optional[List[MagicMock]] = None,
        system_prompt: Optional[str] = None,
    ):
        prompt = MagicMock(spec=llm.Prompt)
        prompt.prompt = prompt_text
        prompt.attachments = attachments or []
        prompt.system = system_prompt
        prompt.options = MagicMock()
        prompt.options.model_dump = mocker.Mock(return_value={})
        return prompt

    return _factory


# Removed custom captured_stderr, will use capsys

# --- Tests for _build_messages (which calls _encode_attachments) ---


def test_build_messages_with_text_and_valid_image(
    vlm_model, mock_prompt_factory, mock_attachment_factory
):
    image_attachment = mock_attachment_factory(
        mime_type="image/png", base64_content="base64_image_data_png"
    )
    prompt = mock_prompt_factory(
        prompt_text="Describe this image.", attachments=[image_attachment]
    )
    messages = vlm_model._build_messages(prompt, conversation=None)

    assert len(messages) == 1
    user_message = messages[0]
    assert user_message["role"] == "user"
    assert isinstance(user_message["content"], list)
    assert len(user_message["content"]) == 2
    assert user_message["content"][0]["type"] == "text"
    assert user_message["content"][0]["text"] == "Describe this image."
    assert user_message["content"][1]["type"] == "image_url"
    assert (
        user_message["content"][1]["image_url"]["url"]
        == "data:image/png;base64,base64_image_data_png"
    )


def test_build_messages_with_only_valid_image(
    vlm_model, mock_prompt_factory, mock_attachment_factory
):
    image_attachment = mock_attachment_factory(
        mime_type="image/jpeg", base64_content="base64_image_data_jpeg"
    )
    prompt = mock_prompt_factory(prompt_text=None, attachments=[image_attachment])
    messages = vlm_model._build_messages(prompt, conversation=None)

    assert len(messages) == 1
    user_message = messages[0]
    assert user_message["role"] == "user"
    assert isinstance(user_message["content"], list)
    assert len(user_message["content"]) == 1
    assert user_message["content"][0]["type"] == "image_url"
    assert (
        user_message["content"][0]["image_url"]["url"]
        == "data:image/jpeg;base64,base64_image_data_jpeg"
    )


def test_build_messages_with_text_and_multiple_valid_images(
    vlm_model, mock_prompt_factory, mock_attachment_factory
):
    image1 = mock_attachment_factory(
        mime_type="image/png", base64_content="img1_data", path="img1.png"
    )
    image2 = mock_attachment_factory(
        mime_type="image/gif", base64_content="img2_data", path="img2.gif"
    )
    prompt = mock_prompt_factory(
        prompt_text="Compare these images.", attachments=[image1, image2]
    )
    messages = vlm_model._build_messages(prompt, conversation=None)

    assert len(messages) == 1
    user_message = messages[0]
    assert user_message["role"] == "user"
    assert isinstance(user_message["content"], list)
    assert len(user_message["content"]) == 3
    assert user_message["content"][0]["type"] == "text"
    assert user_message["content"][0]["text"] == "Compare these images."
    assert user_message["content"][1]["type"] == "image_url"
    assert (
        user_message["content"][1]["image_url"]["url"]
        == "data:image/png;base64,img1_data"
    )
    assert user_message["content"][2]["type"] == "image_url"
    assert (
        user_message["content"][2]["image_url"]["url"]
        == "data:image/gif;base64,img2_data"
    )


def test_build_messages_unsupported_attachment_type_on_vlm(
    vlm_model, mock_prompt_factory, mock_attachment_factory, capsys
):
    pdf_attachment = mock_attachment_factory(
        mime_type="application/pdf", base64_content="pdf_data", path="doc.pdf"
    )
    prompt = mock_prompt_factory(
        prompt_text="Summarize this.", attachments=[pdf_attachment]
    )
    with patch.dict("os.environ", {"LLM_LMSTUDIO_DEBUG": "1"}):
        messages = vlm_model._build_messages(prompt, conversation=None)

    assert len(messages) == 1
    user_message = messages[0]
    assert user_message["role"] == "user"
    assert isinstance(user_message["content"], list)
    assert len(user_message["content"]) == 1
    assert user_message["content"][0]["type"] == "text"
    assert user_message["content"][0]["text"] == "Summarize this."
    captured = capsys.readouterr()
    assert (
        "LMSTUDIO DEBUG: Attachment type application/pdf not in model's supported image types. Skipping doc.pdf."
        in captured.err
    )


def test_build_messages_image_with_non_vlm_model(
    non_vlm_model, mock_prompt_factory, mock_attachment_factory, capsys
):
    image_attachment = mock_attachment_factory(
        mime_type="image/png", base64_content="img_data"
    )
    prompt = mock_prompt_factory(
        prompt_text="What is this?", attachments=[image_attachment]
    )
    with patch.dict("os.environ", {"LLM_LMSTUDIO_DEBUG": "1"}):
        messages = non_vlm_model._build_messages(prompt, conversation=None)

    assert len(messages) == 1
    user_message = messages[0]
    assert user_message["role"] == "user"
    assert isinstance(user_message["content"], list)
    assert len(user_message["content"]) == 1
    assert user_message["content"][0]["type"] == "text"
    assert user_message["content"][0]["text"] == "What is this?"
    captured = capsys.readouterr()
    assert (
        f"LMSTUDIO WARN: Attachments provided, but the selected model '{non_vlm_model.model_id}' may not support images (supports_images=False). Image attachments will likely be ignored by the model."
        in captured.err
    )
    assert (
        f"LMSTUDIO DEBUG: Model {non_vlm_model.model_id} does not support images, but attachment test.png was provided. Ignoring."
        in captured.err
    )


def test_build_messages_attachment_processing_error(
    vlm_model, mock_prompt_factory, mock_attachment_factory, capsys
):
    good_image = mock_attachment_factory(
        mime_type="image/png", base64_content="good_data", path="good.png"
    )
    bad_image = mock_attachment_factory(
        base64_content_raises=IOError("File read error"), path="bad.png"
    )
    prompt = mock_prompt_factory(
        prompt_text="Process these.", attachments=[good_image, bad_image]
    )
    messages = vlm_model._build_messages(prompt, conversation=None)

    assert len(messages) == 1
    user_message = messages[0]
    assert user_message["role"] == "user"
    assert isinstance(user_message["content"], list)
    assert len(user_message["content"]) == 2
    assert user_message["content"][0]["type"] == "text"
    assert user_message["content"][1]["type"] == "image_url"
    assert (
        user_message["content"][1]["image_url"]["url"]
        == "data:image/png;base64,good_data"
    )
    captured = capsys.readouterr()
    assert (
        "LMSTUDIO WARN: Could not process attachment bad.png: File read error. Skipping."
        in captured.err
    )


def test_build_messages_no_text_no_valid_attachments(
    vlm_model, mock_prompt_factory, mock_attachment_factory, capsys
):
    failing_attachment = mock_attachment_factory(
        resolve_type_raises=Exception("Cannot resolve"), path="fail.img"
    )
    prompt = mock_prompt_factory(prompt_text=None, attachments=[failing_attachment])
    with patch.dict("os.environ", {"LLM_LMSTUDIO_DEBUG": "1"}):
        messages = vlm_model._build_messages(prompt, conversation=None)

    assert len(messages) == 1
    user_message = messages[0]
    assert user_message["role"] == "user"
    assert user_message["content"] == ""
    captured = capsys.readouterr()
    assert (
        "LMSTUDIO WARN: Could not process attachment fail.img: Cannot resolve. Skipping."
        in captured.err
    )
    assert (
        "LMSTUDIO DEBUG: Building message for current turn with no text and no encodable images. Sending empty text content."
        in captured.err
    )


def test_build_messages_conversation_history_with_image(
    vlm_model, mock_prompt_factory, mock_attachment_factory, capsys
):
    prev_image_attachment = mock_attachment_factory(
        mime_type="image/jpeg", base64_content="prev_img_data", path="prev.jpg"
    )
    prev_prompt_obj = mock_prompt_factory(
        prompt_text="What was this?", attachments=[prev_image_attachment]
    )
    mock_prev_response = MagicMock()
    mock_prev_response.prompt = prev_prompt_obj
    mock_prev_response.text_or_raise = MagicMock(return_value="It was a cat.")
    mock_conversation = MagicMock()
    mock_conversation.responses = [mock_prev_response]
    current_prompt = mock_prompt_factory(prompt_text="And what color was it?")
    messages = vlm_model._build_messages(current_prompt, conversation=mock_conversation)

    assert len(messages) == 3
    prev_user_message = messages[0]
    assert prev_user_message["role"] == "user"
    assert isinstance(prev_user_message["content"], list)
    assert len(prev_user_message["content"]) == 2
    assert prev_user_message["content"][0]["type"] == "text"
    assert prev_user_message["content"][0]["text"] == "What was this?"
    assert prev_user_message["content"][1]["type"] == "image_url"
    assert (
        prev_user_message["content"][1]["image_url"]["url"]
        == "data:image/jpeg;base64,prev_img_data"
    )
    prev_assistant_message = messages[1]
    assert prev_assistant_message["role"] == "assistant"
    assert prev_assistant_message["content"] == "It was a cat."
    current_user_message = messages[2]
    assert current_user_message["role"] == "user"
    assert isinstance(current_user_message["content"], list)
    assert len(current_user_message["content"]) == 1
    assert current_user_message["content"][0]["type"] == "text"
    assert current_user_message["content"][0]["text"] == "And what color was it?"
    # No specific stderr output is expected in this test case for successful VLM processing
    # captured = capsys.readouterr() # Ensure no unexpected warnings/errors if needed for debugging
    # assert captured.err == "" # Example if strictly no stderr is expected


def test_encode_tools_produces_function_payload(vlm_model):
    tools = [
        llm.Tool(
            name="get_current_time",
            description="Return the current time.",
            input_schema={"type": "object", "properties": {}, "required": []},
        ),
        llm.Tool(
            name="get_weather",
            description="Return mock weather for a city.",
            input_schema={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name to inspect",
                    }
                },
                "required": ["location"],
            },
        ),
    ]

    encoded = vlm_model._encode_tools(tools)

    assert encoded == [
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Return the current time.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Return mock weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name to inspect",
                        }
                    },
                    "required": ["location"],
                },
            },
        },
    ]


def test_build_messages_includes_tool_calls_from_history(
    vlm_model, mock_prompt_factory
):
    conversation = MagicMock()
    prev_prompt = mock_prompt_factory(prompt_text="Please look up the weather.")
    prev_response = MagicMock()
    prev_response.prompt = prev_prompt
    prev_response.text_or_raise = MagicMock(return_value="Calling get_weather now.")
    tool_call = llm.ToolCall(
        name="get_weather",
        arguments={"location": "Berlin"},
        tool_call_id="call_weather_1",
    )
    prev_response.tool_calls_or_raise = MagicMock(return_value=[tool_call])
    conversation.responses = [prev_response]

    current_prompt = mock_prompt_factory(prompt_text="Thanks!")

    messages = vlm_model._build_messages(current_prompt, conversation=conversation)

    assistant_message = next(msg for msg in messages if msg["role"] == "assistant")
    assert assistant_message["content"] == "Calling get_weather now."
    assert "tool_calls" in assistant_message
    assert assistant_message["tool_calls"][0]["id"] == "call_weather_1"
    assert assistant_message["tool_calls"][0]["function"]["name"] == "get_weather"
    assert (
        assistant_message["tool_calls"][0]["function"]["arguments"]
        == '{"location": "Berlin"}'
    )


def test_build_messages_includes_current_tool_results(vlm_model, mock_prompt_factory):
    prompt = mock_prompt_factory(prompt_text="Here are the results.")
    prompt.tool_results = [
        llm.ToolResult(
            name="get_current_time", tool_call_id="call_time_1", output="10:15"
        )
    ]

    messages = vlm_model._build_messages(prompt, conversation=None)

    assert messages[0]["role"] == "tool"
    assert messages[0]["tool_call_id"] == "call_time_1"
    assert messages[0]["content"] == "10:15"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"][0]["type"] == "text"
    assert messages[1]["content"][0]["text"] == "Here are the results."


def test_execute_handles_tool_call_response(monkeypatch, vlm_model):
    monkeypatch.setattr(
        llm_lmstudio.LMStudioModel, "_is_model_loaded", lambda self: True
    )

    api_response = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_weather_123",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "Berlin"}',
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 42, "completion_tokens": 5},
    }

    class FakePostResponse:
        def __init__(self, payload):
            self._payload = payload
            self.text = json.dumps(payload)

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    last_request = {}

    def fake_post(url, json=None, stream=False, timeout=None):
        last_request["url"] = url
        last_request["json"] = json
        return FakePostResponse(api_response)

    monkeypatch.setattr(llm_lmstudio.requests, "post", fake_post)

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

    response = MagicMock(spec=llm.Response)

    result = list(
        vlm_model.execute(
            prompt=prompt,
            stream=False,
            response=response,
            conversation=None,
        )
    )

    assert result == [""]
    response.add_tool_call.assert_called_once()
    tool_call_arg = response.add_tool_call.call_args[0][0]
    assert tool_call_arg.name == "get_weather"
    assert tool_call_arg.arguments == {"location": "Berlin"}
    assert tool_call_arg.tool_call_id == "call_weather_123"
    response.set_usage.assert_called_once_with(input=42, output=5)

    sent_tools = last_request["json"]["tools"]
    assert sent_tools[0]["function"]["name"] == "get_weather"
    assert sent_tools[0]["function"]["parameters"]["required"] == ["location"]
    final_message = last_request["json"]["messages"][-1]
    assert final_message["content"][0]["text"] == "Please check the weather in Berlin."


def test_system_prompt_in_build_messages(vlm_model, mock_prompt_factory):
    prompt = mock_prompt_factory(
        prompt_text="User query.", system_prompt="You are helpful."
    )
    messages = vlm_model._build_messages(prompt, conversation=None)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are helpful."
    assert messages[1]["role"] == "user"


def test_system_prompt_in_conversation_history(vlm_model, mock_prompt_factory):
    prev_prompt_obj = mock_prompt_factory(
        prompt_text="Previous user query.", system_prompt="Be very concise."
    )
    mock_prev_response = MagicMock()
    mock_prev_response.prompt = prev_prompt_obj
    mock_prev_response.text_or_raise = MagicMock(return_value="OK.")
    mock_conversation = MagicMock()
    mock_conversation.responses = [mock_prev_response]
    current_prompt = mock_prompt_factory(prompt_text="Current query.")
    messages = vlm_model._build_messages(current_prompt, conversation=mock_conversation)

    assert len(messages) == 4
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "Be very concise."
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"
    assert messages[3]["role"] == "user"


def test_attempt_load_model_respects_server_list(monkeypatch):
    monkeypatch.setattr(llm_lmstudio, "SERVER_LIST", ["https://10.0.0.5:9000"])
    monkeypatch.setattr(
        llm_lmstudio.LMStudioModel, "_is_model_loaded", lambda self: True
    )

    captured_cmd = {}

    class FakeProcess:
        def __init__(self):
            self.stdout = io.StringIO("")
            self.stderr = io.StringIO("")
            self.returncode = 0

        def wait(self, timeout=None):
            return 0

    def fake_popen(cmd, **kwargs):
        captured_cmd["value"] = cmd
        return FakeProcess()

    monkeypatch.setattr(llm_lmstudio.subprocess, "Popen", fake_popen)

    model = llm_lmstudio.LMStudioModel(
        model_id="test-id",
        base_url="https://10.0.0.5:9000",
        raw_id="test-raw",
        api_path_prefix="/api/v0",
    )

    assert model._attempt_load_model() is True
    assert captured_cmd["value"] == [
        "lms",
        "load",
        "--exact",
        "test-raw",
        "--host",
        "10.0.0.5",
        "--port",
        "9000",
    ]
