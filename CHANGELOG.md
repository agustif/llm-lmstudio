# Changelog

## Unreleased


## v0.3.0 - 2026-08-11

### Added
- Support for LLM 0.32 structured messages and streaming events, including structured text, reasoning, and tool-call response parts.
- Resolved model IDs and raw provider response payloads are now recorded on responses when LM Studio supplies them.

### Changed
- Conversation requests now consume the canonical `prompt.messages` chain instead of rebuilding history from `conversation.responses`.
- Raised the minimum LLM version to 0.32.
- Raised the minimum LM Studio version to 0.4.0, which introduced the native v1 REST API.
- Automatic model loading now uses `POST /api/v1/models/load` instead of the `lms` CLI.
- Automatic loading no longer supports `LLM_LMSTUDIO_TTL`. The new load endpoint does not accept a TTL parameter.

### Fixed
- Automatic loading now works when an API model ID differs from its full model path.
- Model loading no longer writes CLI terminal control sequences to stderr.
- System prompts are no longer repeated between tool-call rounds.
- Assistant messages containing tool calls but no text are preserved with `content: null`, keeping tool-call and tool-result ordering valid.
- Streaming requests continue through final usage-only chunks and record usage consistently in sync and async flows.

### Known Issues
- Authenticated remote LM Studio servers are not supported. Community contributions are welcome.


## v0.2.1 - 2025-12-06

### Changed
- If LM Studio is not running, only show a warning about that when `LLM_LMSTUDIO_DEBUG=1`. This allows having the plugin installed even if you do not use LM Studio models all the time.

## v0.2.0 - 2025-12-03

### Added
- Full tool-calling support in both sync and async flows, including encoding of `prompt.tools`, replaying prior tool results, and surfacing streamed tool invocations back through `llm.Response`.
- `shell.nix` based setup of development environment
- End to end tests to validate plugin.

### Changed
- All discovered LM Studio models are now registered as `lmstudio/<model_id>` (or `lmstudio@host/...` when multiple servers are configured) so CLI usage is consistent across environments.
- Migrated the package to a `src/` layout, added an explicit `[build-system]` table, generated a pinned `uv.lock`, refreshed README instructions to favor `uv run --all-extras pytest`.

### Fixed
- Automatic `lms load` calls now pass `--exact`, `--host`, `--port` from  `LMSTUDIO_API_BASE`, and optional `--ttl $LLM_LMSTUDIO_TTL`, show clearer progress, and use longer timeouts to make cold starts and long responses more reliable.

### Breaking Changes
- Raised `requires-python` to `>=3.10`, dropping support for Python 3.8 and 3.9 as [3.9 has reached end of life by by 2025-10-31](https://endoflife.date/python).
- Scripts referencing bare LM Studio IDs must be updated to the new `lmstudio/...` prefix format.

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
