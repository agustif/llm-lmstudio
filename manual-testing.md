# Manual testing against LM Studio

*2026-08-11T08:48:12Z by Showboat 0.6.1*
<!-- showboat-id: 6416c594-b38c-4ec6-95e6-f3b92429c454 -->

This session tests llm-lmstudio against LM Studio 0.4.20 or newer. It uses minicpm5-1b as the small GGUF model and prism-ml/bonsai-27b as the smallest installed MLX model.

The commands use the project virtual environment directly because this document must also work in non-interactive shells.

## Environment and model discovery

```bash
.venv/bin/llm --version
```

```output
llm, version 0.32
```

```bash
.venv/bin/llm models list | rg '^lmstudio/(minicpm5-1b|prism-ml/bonsai-27b) ' | sed 's/ ●//' | sort
```

```output
lmstudio/minicpm5-1b ⚒
lmstudio/prism-ml/bonsai-27b 👁 ⚒
```

The model list contains both test models. LM Studio reports vision and tool capabilities with display suffixes.

```bash
.venv/bin/llm models list --options -m lmstudio/minicpm5-1b | sed '1s/ ●//'
```

```output
lmstudio/minicpm5-1b ⚒
  Options:
    temperature: float
      Sampling temperature
    top_p: float
      Nucleus sampling
    max_tokens: int
      Maximum tokens
    stop: array
      Stop sequences
  Attachment types:
    image/gif, image/jpeg, image/png, image/webp
  Features:
  - streaming
  - schemas
  - tools
  - async
```

```bash
.venv/bin/llm models list --options -m lmstudio/prism-ml/bonsai-27b | sed '1s/ ●//'
```

```output
lmstudio/prism-ml/bonsai-27b 👁 ⚒
  Options:
    temperature: float
      Sampling temperature
    top_p: float
      Nucleus sampling
    max_tokens: int
      Maximum tokens
    stop: array
      Stop sequences
  Attachment types:
    image/gif, image/jpeg, image/png, image/webp
  Features:
  - streaming
  - schemas
  - tools
  - async
```

## Automatic loading and basic prompting

This check requests a GGUF model unload and verifies the unloaded state. The next request must load it through `POST /api/v1/models/load` before generation.

```bash
lms unload minicpm5-1b >/dev/null 2>&1 || true
if .venv/bin/llm models list | grep -Fq 'lmstudio/minicpm5-1b ●'; then
    echo 'GGUF model is still loaded' >&2
    exit 1
fi
.venv/bin/llm -m lmstudio/minicpm5-1b -R "Reply with exactly: GGUF_OK" 2>&1 | sed '/^[[:space:]]*$/d'
```

```output
GGUF_OK
```

This check repeats the same path with the installed MLX model. LM Studio can report `Model Not Found` when the model is already unloaded.

```bash
lms unload prism-ml/bonsai-27b >/dev/null 2>&1 || true
if .venv/bin/llm models list | grep -Fq 'lmstudio/prism-ml/bonsai-27b ●'; then
    echo 'MLX model is still loaded' >&2
    exit 1
fi
.venv/bin/llm -m lmstudio/prism-ml/bonsai-27b -R "Reply with exactly: MLX_OK" 2>&1 | sed '/^[[:space:]]*$/d'
```

```output
MLX_OK
```

## Prompt modes and structured output

```bash
set -o pipefail; .venv/bin/llm -m lmstudio/minicpm5-1b --no-stream -R -s "Answer with one word only." "What is the capital of France?" | grep -ixq 'Paris' && echo 'basic prompt: OK'
```

```output
basic prompt: OK
```

```bash
set -o pipefail; .venv/bin/llm -m lmstudio/minicpm5-1b -R --schema 'name, age int' 'Invent one dog. Use a short name.' | jq -e 'type == "object" and keys == ["age", "name"] and (.name | type == "string") and (.age | type == "number") and (.age == (.age | floor))' >/dev/null && echo 'schema shape: OK'
```

```output
schema shape: OK
```

## Tool calling

```bash
set -o pipefail; .venv/bin/llm -m lmstudio/prism-ml/bonsai-27b -R --functions $'def multiply(x: int, y: int) -> int:\n    """Multiply two integers."""\n    return x * y' --td 'Use the tool to calculate 123 * 456. Give only the result.' 2>&1 | awk '/Tool call: multiply/ { tool = 1 } /56088/ { result = 1 } END { if (!tool || !result) exit 1; print "tool call: OK" }'
```

```output
tool call: OK
```

## Conversation continuation

```bash
set -o pipefail; .venv/bin/llm -m lmstudio/prism-ml/bonsai-27b -R 'Remember the code word cobalt. Reply with exactly: OK' | grep -ixq 'OK' && echo 'conversation started: OK'
```

```output
conversation started: OK
```

```bash
set -o pipefail; .venv/bin/llm -c -R 'What code word did I ask you to remember? Reply with one word.' | grep -ixq 'cobalt' && echo 'conversation recall: OK'
```

```output
conversation recall: OK
```

## Embeddings

```bash
.venv/bin/llm embed-models | rg '^LMStudioEmbeddingModel:' | sort
```

```output
LMStudioEmbeddingModel: text-embedding-jina-embeddings-v4-text-retrieval
LMStudioEmbeddingModel: text-embedding-mxbai-embed-large-v1
LMStudioEmbeddingModel: text-embedding-nomic-embed-text-v1.5@q4_k_m
LMStudioEmbeddingModel: text-embedding-nomic-embed-text-v1.5@q8_0
LMStudioEmbeddingModel: text-embedding-qwen3-vl-embedding-2b
LMStudioEmbeddingModel: text-embedding-qwen3-vl-reranker-2b
```

The base64 length gives a concise check that the embedding endpoint returned a non-empty vector.

```bash
set -o pipefail; .venv/bin/llm embed -m text-embedding-nomic-embed-text-v1.5@q4_k_m -c pelican -f base64 | tr -d '\n' | wc -c | grep -Eq '^[[:space:]]*4096$' && echo 'embedding: OK'
```

```output
embedding: OK
```

## Automated test suite

```bash
.venv/bin/pytest -q >/dev/null && echo 'tests passed'
```

```output
tests passed
```

## Async Python acceptance test

This live Python test covers the async integration boundary. It uses GGUF for streaming, usage, and structured output, then MLX for a tool chain.

```.venv/bin/python
import asyncio
import json

import llm


async def main():
    gguf = llm.get_async_model("lmstudio/minicpm5-1b")
    print("gguf async:", isinstance(gguf, llm.AsyncModel))

    response = gguf.prompt(
        "Reply with one short greeting.",
        hide_reasoning=True,
    )
    chunks = [chunk async for chunk in response]
    text = "".join(chunks).strip()
    assert text
    print("stream: OK")

    usage = await response.usage()
    assert usage.input is not None and usage.input > 0
    assert usage.output is not None and usage.output > 0
    print("usage recorded:", usage.input > 0 and usage.output > 0)

    schema_text = await gguf.prompt(
        "Return a dog named Pip who is 4 years old.",
        schema=llm.schema_dsl("name, age int"),
        hide_reasoning=True,
    ).text()
    schema_result = json.loads(schema_text)
    assert set(schema_result) == {"name", "age"}
    assert isinstance(schema_result["name"], str)
    assert isinstance(schema_result["age"], int)
    print("schema: OK")

    mlx = llm.get_async_model("lmstudio/prism-ml/bonsai-27b")
    print("mlx async:", isinstance(mlx, llm.AsyncModel))

    def multiply(x: int, y: int) -> int:
        """Multiply two integers."""
        return x * y

    result = await mlx.chain(
        "Use the tool to calculate 123 * 456. Give only the result.",
        tools=[multiply],
        hide_reasoning=True,
    ).text()
    assert "56088" in result
    print("tool result: OK")


asyncio.run(main())

```

```output
gguf async: True
stream: OK
usage recorded: True
schema: OK
mlx async: True
tool result: OK
```

## Vision attachment

This check uses a local 320 px copy of Crested Tern Tasmania (edit).jpg from Wikimedia Commons.

```bash
set -o pipefail; .venv/bin/llm -m lmstudio/prism-ml/bonsai-27b -R -a manual-testing-assets/crested-tern.jpg 'Is the primary subject of this image a bird? Reply with exactly YES or NO.' | grep -ixq 'YES' && echo 'vision: OK'
```

```output
vision: OK
```

## Synchronous streaming usage

This live check verifies that a synchronous streamed response includes positive input and output token counts.

```bash
set -o pipefail; .venv/bin/llm -m lmstudio/prism-ml/bonsai-27b -R -u 'Reply with one short greeting.' 2>&1 | awk '/Token usage: [1-9][0-9]* input, [1-9][0-9]* output/ { usage = 1 } END { if (!usage) exit 1; print "streaming usage: OK" }'
```

```output
streaming usage: OK
```
