# Manual testing against LM Studio

Manual test session for the LLM 0.32 upgrade, run against `qwen/qwen3.5-9b` served
locally by LM Studio. LLM version 0.32.

Reasoning traces are streamed to standard error by LLM 0.32, so most commands below
use `-R/--hide-reasoning` to keep the output readable. Where reasoning is relevant it
is shown.

## Model registration

```bash
uv run llm models list | grep -i lmstudio
```

> lmstudio/meta/muse-glimmer ● 👁 ⚒
> lmstudio/qwen/qwen3.5-9b ● 👁 ⚒
> lmstudio/qwen/qwen3.5-9b:2 ● 👁 ⚒
> lmstudio/ternary-bonsai-27b 👁 ⚒
> lmstudio/bonsai-27b-mlx 👁 ⚒
> lmstudio/ornith-1.0-35b ⚒
> lmstudio/google/gemma-4-12b-qat 👁 ⚒
> lmstudio/north-mini-code-1.0 ⚒
> lmstudio/google/gemma-4-12b 👁 ⚒
> lmstudio/granite-4.1-30b ⚒
> lmstudio/granite-4.1-8b-fp8 ⚒
> lmstudio/granite-4.1-3b ⚒
> lmstudio/qwen3.6-35b-a3b 👁 ⚒
> lmstudio/minimax-m2.7 ⚒
> lmstudio/google/gemma-4-26b-a4b 👁 ⚒
> lmstudio/google/gemma-4-31b 👁 ⚒
> lmstudio/google/gemma-4-e4b 👁 ⚒
> lmstudio/google/gemma-4-e2b 👁 ⚒
> lmstudio/qwen3.5-27b-claude-4.6-opus-reasoning-distilled-v2 👁 ⚒
> lmstudio/openai/gpt-oss-120b ⚒
> lmstudio/qwen/qwen3-coder-next ⚒
> lmstudio/qwen3.5-4b 👁 ⚒
> lmstudio/qwen/qwen3.5-35b-a3b 👁 ⚒

Options and advertised features for the model under test:

```bash
uv run llm models list --options -m lmstudio/qwen/qwen3.5-9b
```

> lmstudio/qwen/qwen3.5-9b ● 👁 ⚒
>
> &nbsp;&nbsp;Options:
>
> &nbsp;&nbsp;&nbsp;&nbsp;temperature: float
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Sampling temperature
>
> &nbsp;&nbsp;&nbsp;&nbsp;top_p: float
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Nucleus sampling
>
> &nbsp;&nbsp;&nbsp;&nbsp;max_tokens: int
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Maximum tokens
>
> &nbsp;&nbsp;&nbsp;&nbsp;stop: array
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Stop sequences
>
> &nbsp;&nbsp;Attachment types:
>
> &nbsp;&nbsp;&nbsp;&nbsp;image/gif, image/jpeg, image/png, image/webp
>
> &nbsp;&nbsp;Features:
>
> &nbsp;&nbsp;- streaming
> &nbsp;&nbsp;- schemas
> &nbsp;&nbsp;- tools
> &nbsp;&nbsp;- async

## Basic prompting

```bash
uv run llm -m lmstudio/qwen/qwen3.5-9b -R 'What is the capital of France? One word.'
```

> Paris

A system prompt, with streaming disabled:

```bash
uv run llm -m lmstudio/qwen/qwen3.5-9b --no-stream -R -s 'You are a pirate. Always answer in pirate speak.' 'What is 2+2?'
```

> Arrr! Avast ye, matey! Two plus two be four, like four doubloons on the deck!

## Reasoning

Without `-R`, reasoning is streamed to standard error, so it does not interfere with
piping the response on standard output. Here standard error is discarded:

```bash
uv run llm -m lmstudio/qwen/qwen3.5-9b 'Invent a haunted house' --schema 'name, year_built int, one_sentence_bio' 2>/dev/null
```

> { "name": "The House of Unfinished Sentences","year_built": 1893,"one_sentence_bio": "A Victorian manor where words that were never spoken become trapped in the plaster, and if you speak too loudly, the house will swallow your voice to replace it with its own." }

Reasoning is stored as its own part in the 0.32 log schema, under a `## Reasoning`
heading, separate from `## Response`:

```bash
uv run llm logs -c -n 1
```

> \# 2026-08-10T19:53:48    conversation: 01kzpkt270pbvzk71abqtq3svw id: 01kzpkt271ecnjs6m565jt1z1x
>
> Model: **lmstudio/qwen/qwen3.5-9b** (resolved: **qwen/qwen3.5-9b**)
>
> \## Prompt
>
> My favourite animal is the pelican. Remember that. Reply with just "OK".
>
> \## Reasoning
>
> Thinking Process:
>
> 1.  **Analyze the Request:**
>     *   User statement: "My favourite animal is the pelican. Remember that."
>     *   Constraint: "Reply with just 'OK'."
>     *   Goal: Acknowledge the information without adding extra text, adhering strictly to the constraint.
>
> 2.  **Check Constraints:**
>     *   Must reply *only* with "OK".
>     *   No explanations, no greetings, no confirmations beyond that specific string.
>
> 3.  **Formulate Output:**
>     *   String: "OK"
>
> 4.  **Final Review:**
>     *   Does it meet the constraint? Yes.
>     *   Is it accurate to the instruction? Yes.
>
> 5.  **Execute:** Output "OK".
>
> \## Response
>
> OK

## Schemas

Using the concise schema syntax:

```bash
uv run llm -m lmstudio/qwen/qwen3.5-9b 'Invent a cool dog' --schema 'name, age int, one_sentence_bio' 2>/dev/null
```

> {
>   "name": "The Spectre Hound",
>   "age": 3,
>   "one_sentence_bio": "A sleek, cyberpunk-inspired breed with adaptive fur and silent-paw capabilities designed for high-tech urban environments."
> }

Multiple items with `--schema-multi`:

```bash
uv run llm -m lmstudio/qwen/qwen3.5-9b --schema-multi 'name, species, weight_kg float' 'Invent 3 pelicans' 2>/dev/null
```

> {
>     "items": [
>         {
>             "name": "Barnaby the Bottomless",
>             "species": "Great White Pelican (Modified)",
>             "weight_kg": 1450.5
>         },
>         {
>             "name": "Inkwell the Messenger",
>             "species": "Pacific Pink-backed Pelican",
>             "weight_kg": 850.2
>         },
>         {
>             "name": "Sonar-Beak Whistler",
>             "species": "Horned Pelican (Fictional Variant)",
>             "weight_kg": 620.1
>         }
>     ]
> }

## Attachments

An image fetched from a URL:

```bash
uv run llm -m lmstudio/qwen/qwen3.5-9b -R 'Describe this image in two sentences.' -a https://static.simonwillison.net/static/2025/two-pelicans.jpg
```

> Two brown pelicans soar through a clear blue sky with their massive wings fully spread to catch the air. One bird flies slightly lower in the foreground, while its companion glides just above it towards the right side of the frame.

The same image from a local file, exercising the base64 path:

```bash
curl -so /tmp/two-pelicans.jpg https://static.simonwillison.net/static/2025/two-pelicans.jpg
uv run llm -m lmstudio/qwen/qwen3.5-9b -R -a /tmp/two-pelicans.jpg 'What species of bird is this? Two words.'
```

> Brown Pelican

## Conversations with `llm -c`

```bash
uv run llm -m lmstudio/qwen/qwen3.5-9b -R 'My favourite animal is the pelican. Remember that. Reply with just "OK".'
```

> OK

```bash
uv run llm -c -R 'What is my favourite animal? Answer in one word.'
```

> Pelican

```bash
uv run llm -c -R 'Now name three facts about that animal, very briefly.'
```

> 1. Massive throat pouch traps fish.
> 2. Wingspans often exceed two meters.
> 3. Beaks can hold over 10 liters of water.

## Tools

Two of the default tools that ship with LLM, called in sequence within a single
prompt:

```bash
uv run llm -m lmstudio/qwen/qwen3.5-9b -R --tool llm_version --tool llm_time 'What version of LLM is installed, and what time is it? Use your tools.' --td
```

> Tool call: llm_version({})
> &nbsp;&nbsp;0.32
>
> Tool call: llm_time({})
> &nbsp;&nbsp;{
> &nbsp;&nbsp;&nbsp;&nbsp;"utc_time": "2026-08-10 19:54:35 UTC",
> &nbsp;&nbsp;&nbsp;&nbsp;"utc_time_iso": "2026-08-10T19:54:35.714014+00:00",
> &nbsp;&nbsp;&nbsp;&nbsp;"local_timezone": "PDT",
> &nbsp;&nbsp;&nbsp;&nbsp;"local_time": "2026-08-10 12:54:35",
> &nbsp;&nbsp;&nbsp;&nbsp;"timezone_offset": "UTC-7:00",
> &nbsp;&nbsp;&nbsp;&nbsp;"is_dst": true
> &nbsp;&nbsp;}
>
> Here's the information you requested:
>
> **LLM Version:** 0.32
>
> **Current Time:**
> - **Local Time (PDT):** August 10, 2026 at 12:54:35
> - **UTC Time:** August 10, 2026 at 19:54:35 UTC
> - **Timezone Offset:** UTC-7:00
> - **Daylight Saving Time:** Currently active

A custom Python function tool:

```bash
uv run llm -m lmstudio/qwen/qwen3.5-9b -R --functions '
def multiply(x: int, y: int) -> int:
    """Multiply two numbers."""
    return x * y
' 'what is 34234 * 213345?' --td
```

> Tool call: multiply({'x': 34234, 'y': 213345})
> &nbsp;&nbsp;7303652730
>
> 34234 × 213345 = 7,303,652,730

A longer tool chain against a SQLite database, which also exercises tool errors being
fed back to the model and parallel tool calls within a single response. The database
was created with four pelicans, with a `weight_kg` column:

```bash
uv run llm -m lmstudio/qwen/qwen3.5-9b -R --functions '
import sqlite3

def list_tables() -> str:
    "List tables in the pelicans database"
    db = sqlite3.connect("/tmp/pelicans.db")
    return str(db.execute("select name from sqlite_master where type=\"table\"").fetchall())

def query(sql: str) -> str:
    "Run a SQL SELECT query against the pelicans database and return results"
    db = sqlite3.connect("/tmp/pelicans.db")
    return str(db.execute(sql).fetchall())
' 'How many pelicans are in the database, and which one is heaviest? Explore the schema first.' --td --cl 8
```

> Tool call: query({'sql': 'SELECT id, name, species, weight FROM pelicans ORDER BY weight DESC LIMIT 1;'})
> &nbsp;&nbsp;Error: no such column: weight
> &nbsp;&nbsp;Exception: no such column: weight
>
> Tool call: query({'sql': 'SELECT * FROM pelicans LIMIT 1;'})
> &nbsp;&nbsp;[(1, 'Percy', 'Brown Pelican', 3.6)]
>
> Tool call: query({'sql': 'PRAGMA table_info(pelicans);'})
> &nbsp;&nbsp;[(0, 'id', 'INTEGER', 0, None, 1), (1, 'name', 'TEXT', 0, None, 0), (2, 'species', 'TEXT', 0, None, 0), (3, 'weight_kg', 'REAL', 0, None, 0)]
>
> Tool call: query({'sql': 'SELECT id, name, species, weight_kg FROM pelicans ORDER BY weight_kg DESC LIMIT 1;'})
> &nbsp;&nbsp;[(4, 'Penny', 'Dalmatian Pelican', 12.8)]
>
> Based on my exploration of the database schema, here's what I found:
>
> \## Database Schema
> The `pelicans` table has the following structure:
> - **id** (INTEGER) - Unique identifier
> - **name** (TEXT) - Pelican name
> - **species** (TEXT) - Pelican species type
> - **weight_kg** (REAL) - Weight in kilograms
>
> \## Answers to Your Questions
>
> \### 1. How many pelicans are in the database?
> **4 pelicans** are stored in the database.
>
> \### 2. Which one is heaviest?
> The heaviest pelican is **Penny**, a **Dalmatian Pelican** weighing **12.8 kg**.

Tool results carry across a conversation continuation. First a tool call:

```bash
uv run llm -m lmstudio/qwen/qwen3.5-9b -R --tool llm_time 'What time is it? Use the tool.' --td
```

> **UTC Time:** August 10, 2026 at 8:04:10 PM
>
> Yes, daylight saving time is currently in effect.

Then a follow-up that depends on the earlier tool result:

```bash
uv run llm -c -R 'What timezone did you just report? One word.'
```

> PDT

## Token usage

This uncovered a bug, fixed in this branch. LM Studio only emits the final usage
chunk when the request sets `stream_options: {"include_usage": true}`, which the
plugin was not sending, so streamed responses recorded no token counts at all.

Before the fix, streamed responses reported nothing:

```bash
uv run llm -m lmstudio/qwen/qwen3.5-9b -R -u 'Count to 3'
```

> 1, 2, 3Token usage:

After the fix:

```bash
uv run llm -m lmstudio/qwen/qwen3.5-9b -R -u 'Count to 3'
```

> 1, 2, 3Token usage: 14 input, 251 output, {"completion_tokens_details": {"reasoning_tokens": 241}}

Streaming with tools, which also reported nothing before the fix:

```bash
uv run llm -m lmstudio/qwen/qwen3.5-9b -R -u --tool llm_version 'What llm version? Use the tool.'
```

> Token usage: 362 input, 45 output, {"completion_tokens_details": {"reasoning_tokens": 30}}

Non-streaming was unaffected and still works:

```bash
uv run llm -m lmstudio/qwen/qwen3.5-9b -R -u --no-stream 'Count to 3'
```

> Token usage: 14 input, 307 output, {"completion_tokens_details": {"reasoning_tokens": 297}}
>
> 1, 2, 3

Schemas force `stream=false` internally, so these already reported usage:

```bash
uv run llm -m lmstudio/qwen/qwen3.5-9b -R -u --schema 'n int' 'pick a number'
```

> { "n": 42 }Token usage: 13 input, 1,216 output, {"completion_tokens_details": {"reasoning_tokens": 1184}}

## Async

The async model was exercised with a script covering streaming, schemas, tools,
conversations, attachments and usage:

```bash
uv run python async_test.py
```

> model: lmstudio/qwen/qwen3.5-9b | async: True
> streamed chunks: 2 -> Blue Jay
> schema: {
>     "name": "The Prism-Pelican (*Pelecanus refracta*)",
>     "weight_kg": 8.5
> }
> tool result: The wingspan of a pelican is 3.5 metres.
> conversation recall: Percy
> vision: 2
> usage: Usage(input=18, output=1906, details={'completion_tokens_details': {'reasoning_tokens': 1901}})

The script:

```python
import asyncio, llm

async def main():
    model = llm.get_async_model("lmstudio/qwen/qwen3.5-9b")
    print("model:", model.model_id, "| async:", isinstance(model, llm.AsyncModel))

    # 1. streaming async text
    resp = model.prompt("Name one bird. Two words max.", hide_reasoning=True)
    chunks = [c async for c in resp]
    print("streamed chunks:", len(chunks), "->", "".join(chunks).strip())

    # 2. async schema
    r2 = await model.prompt(
        "Invent a pelican",
        schema=llm.schema_dsl("name, weight_kg float"),
        hide_reasoning=True,
    ).text()
    print("schema:", r2.strip())

    # 3. async tool calling
    def wingspan(name: str) -> str:
        "Return the wingspan of a bird in metres"
        return "3.5 metres"

    chain = model.chain("What is the wingspan of a pelican? Use the tool.",
                        tools=[wingspan], hide_reasoning=True)
    text = await chain.text()
    print("tool result:", text.strip()[:120])

    # 4. async conversation memory
    conv = model.conversation()
    await conv.prompt("My bird is named Percy. Reply 'OK'.", hide_reasoning=True).text()
    out = await conv.prompt("What is my bird's name? One word.", hide_reasoning=True).text()
    print("conversation recall:", out.strip())

    # 5. async attachment
    r5 = await model.prompt(
        "How many birds in this image? Answer with a digit only.",
        attachments=[llm.Attachment(url="https://static.simonwillison.net/static/2025/two-pelicans.jpg")],
        hide_reasoning=True,
    ).text()
    print("vision:", r5.strip())

    usage = await resp.usage()
    print("usage:", usage)

asyncio.run(main())
```

## Test suite

```bash
uv run pytest -q
```

> .........................                                                [100%]
> 25 passed in 0.78s
