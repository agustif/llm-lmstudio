# llm-lmstudio

This is a plugin for [Simon Willison's LLM command-line utility](https://llm.datasette.io/) that lets you talk to models running on a local [LMStudio](https://lmstudio.ai/) server.

## Installation

Make sure you have `llm` installed, then install this plugin:

```bash
llm install llm-lmstudio
```

## Usage

First, you need LMStudio running with a model loaded. The plugin talks to the LMStudio server API, which usually runs at `http://localhost:1234`.

Once the server is up, the plugin should automatically find the models you have loaded in LMStudio. You can check this using the `llm models` command:

```bash
llm models list
```
You should see your LMStudio models listed, likely prefixed with `lmstudio/`.

To run a prompt against a model:

```bash
# Replace 'your-model-id' with the actual ID shown in 'llm models list'
# e.g., lmstudio/phi-2.Q4_K_S.gguf
llm -m lmstudio/your-model-id 'Write a short poem about a coffee cup'
```

To start an interactive chat session:

```bash
llm chat -m lmstudio/your-model-id
```

You can exit the chat by typing `exit` or `quit`.

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

The plugin connects to the LMStudio server API, expecting it at `http://localhost:1234` by default.

If your LMStudio server is running on a different address or port, you can set the `LMSTUDIO_API_BASE` environment variable:

```bash
export LMSTUDIO_API_BASE="http://your-server-address:port"
```

The plugin will use this address to connect to the `/v1/models`, `/v1/completions`, and `/v1/chat/completions` endpoints.

## Model Options

You can pass generation options supported by the LMStudio API (like `temperature`, `max_tokens`, `top_p`, `stop`) using the `-o` flag:

```bash
llm -m lmstudio/your-model-id -o temperature 0.7 -o max_tokens 100 "Tell me a joke"
```

## Missing features:

- [ ] Add image support on chat for local multi-modal or vision-language models.

~ Almost one-shotted by o3, use at your own risk.
