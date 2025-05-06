# Changelog

## v0.1.0 - 2025-05-07

### Added
- Initial support for Vision Language Models (VLMs) allowing image attachments with prompts via `llm -a path/to/image.png ...`.
- Plugin automatically detects and registers models from local LM Studio server(s).
- Supports LM Studio API `/v1` and newer `/api/v0` for model discovery and interaction.
- `LMStudioModel` and `LMStudioAsyncModel` now include a `display_suffix` (e.g., "● 👁️ ⚒️") to indicate loaded status, vision support, and schema/tool support, visible in `llm inspect <model_id>`.
- Asynchronous operations and testing using `pytest-asyncio` and `pytest-vcr`.
- VCR cassettes for asynchronous tests to enable CI and offline testing.
- `LMSTUDIO_API_BASE` environment variable can now take a comma-separated list of server URLs.
- Automatic model loading attempt via `lms load <model_id>` if the target model isn't loaded (requires `lms` CLI).
- Debug logging controlled by `LLM_LMSTUDIO_DEBUG=1`.
- Support for embedding models.

### Fixed
- Correct construction of OpenAI-compatible payloads for messages, including base64 encoded images.
- Handling of system prompts for both current turn and conversation history.
- Resolved various test failures to achieve a stable test suite.
- Ensured `pytest-vcr` correctly records and replays interactions for async tests.
- Model ID handling for reliable lookup (`llm -m <model_id>`).
- `llm.Response.text()` is now correctly awaited in async tests.

### Changed
- Updated README with improved installation instructions, usage examples, vision support details, configuration, and a development section including VCR cassette generation.

### Known Issues
- Icons in `display_suffix` may not render correctly in all terminals when using `llm models list` but are usually visible with `llm inspect`. 