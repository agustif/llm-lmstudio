# llm-lmstudio

[![PyPI](https://img.shields.io/pypi/v/llm-lmstudio.svg)](https://pypi.org/project/llm-lmstudio/)
[![Changelog](https://img.shields.io/github/v/release/agustif/llm-lmstudio?include_prereleases&label=changelog)](https://github.com/agustif/llm-lmstudio/releases)
[![Tests](https://github.com/agustif/llm-lmstudio/actions/workflows/test.yml/badge.svg)](https://github.com/agustif/llm-lmstudio/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/agustif/llm-lmstudio/blob/main/LICENSE)

This is a plugin for [Simon Willison's LLM command-line utility](https://llm.datasette.io/) that lets you talk to models running on a local [LMStudio](https://lmstudio.ai/) server.

## Installation

Make sure you have `llm` installed, then install this plugin from PyPI:

```bash
pip install llm-lmstudio
```

Or, to install the latest development version directly from GitHub:
```bash
llm install -U git+https://github.com/agustif/llm-lmstudio.git
```
Alternatively, `llm install llm-lmstudio` will also find and install the plugin.

## Usage

First, you need LMStudio running with a model loaded. The plugin talks to the LMStudio server API, which usually runs at `http://localhost:1234`.

Once the server is up, the plugin should automatically find the models you have loaded in LMStudio. You can check this using the `llm models` command:

```bash
llm models list
```
You should see your LMStudio models listed, likely prefixed with `lmstudio/` (e.g., `LMStudio: lmstudio/llava-v1.5-7b`).

To run a prompt against a model:

```bash
# Replace 'your-model-id' with the actual ID shown in 'llm models list'
# e.g., llm -m lmstudio/llava-v1.5-7b "Describe this image"
llm -m lmstudio/your-model-id -o temperature 0.7 -o max_tokens 100 "Tell me a joke"
```

To start an interactive chat session:

```bash
llm chat -m lmstudio/your-model-id
```

You can exit the chat by typing `exit` or `quit`.

### Vision Model Support

The plugin supports vision-language models (VLMs).
- When using a VLM, you can attach images using the standard `llm` attachment syntax:
  ```bash
  llm chat -m lmstudio/your-vlm-id -a path/to/image.png "Describe this image"
  ```
  Or for a single prompt:
  ```bash
  llm -m lmstudio/your-vlm-id -a path/to/image.png "What is in this picture?"
  ```
- The plugin will encode the image and send it to the model.
- This feature's success depends on the specific VLM, its configuration in LM Studio, and LM Studio's API correctly handling image data.
- Models that support vision may have a `👁️` (eye icon) in their `display_suffix` when inspected via `llm inspect -m lmstudio/your-vlm-id`, though this may not always render in `llm models list`.

### Embedding Models

If you have embedding models loaded in LMStudio (their names usually contain "embed"), the plugin will register them too. You can list them with:

```bash
llm embed-models
```

You should see models like `lmstudio/text-embedding-nomic-embed-text-v1.5@f16`.

To generate embeddings for text using one of these models:

```bash
llm embed -m lmstudio/your-embedding-model-id -c "This is the text to embed"
```

## Configuration

The plugin connects to the LMStudio server API. By default, it tries `http://localhost:1234`.

You can configure the server URL(s) using the `LMSTUDIO_API_BASE` environment variable.
- For a single server:
  ```bash
  export LMSTUDIO_API_BASE="http://your-server-address:port"
  ```
- For multiple servers (the plugin will try them in order):
  ```bash
  export LMSTUDIO_API_BASE="http://server1:1234,http://server2:5678,https://server3:custom_port"
  ```
The variable accepts one or more `http[s]://host:port` values, separated by commas (spaces around commas are optional). The plugin automatically attempts to append `/v1` or `/api/v0` to the determined base URL(s) as needed when probing the server.

## Model Options

You can pass generation options supported by the LMStudio API (like `temperature`, `max_tokens`, `top_p`, `stop`) using the `-o` flag:

```bash
llm -m lmstudio/your-model-id -o temperature 0.7 -o max_tokens 100 "Tell me a joke"
```

## Automatic Model Loading

If a selected model is not currently loaded in LM Studio, the plugin will attempt to automatically load it by invoking `lms load <model_id>` (if `lms` CLI is installed and configured). You may see progress messages from LM Studio in your terminal during this process.

## Development

To set up this plugin for development:
1. Clone the repository.
2. Create a virtual environment: `python -m venv .venv`
3. Activate it: `source .venv/bin/activate`
4. Install dependencies, including test dependencies: `pip install -e ".[test]"`
5. Run tests: `pytest` or `PYTHONPATH=. pytest -v`

The asynchronous tests in `tests/test_llm_lmstudio_async.py` use `pytest-vcr` to record and replay HTTP interactions with the LM Studio server. To record new cassettes:
1. Ensure LM Studio is running with the target model(s) loaded (e.g., `llava-v1.5-7b`).
2. Temporarily set `record_mode='all'` in the `@pytest.mark.vcr` decorator for the relevant tests in `tests/test_llm_lmstudio_async.py`.
3. Run `PYTHONPATH=. pytest -v -s tests/test_llm_lmstudio_async.py`.
4. Cassettes will be saved in `tests/cassettes/`.
5. **Important:** Change `record_mode` back to `'once'` after recording.
6. Commit the new/updated cassettes.

## Missing features / Known Issues:

- [x] Add image support on chat for local multi-modal or vision-language models. *(Initial support added. Further testing and refinement are needed for robustness across various VLMs and LM Studio versions.)*
- The reliability and capabilities of image support can vary significantly based on the specific VLM and its implementation within LM Studio.
- The icons indicating model capabilities (loaded, vision, tools) in `llm models list` may not render correctly in all terminals or may be stripped by `llm`. They are more reliably viewed with `llm inspect <model_id>`.