# Section 1 — LangChain Foundations

> **Future with Shivank — AI Harness** · prompt templates, chat models, chains, local models and tracing

## What you'll learn

- How to keep API keys out of your code with a `.env` file
- The three building blocks used in every later section: **PromptTemplate**, **chat model**, and the **chain** (`prompt | llm`)
- How to swap a paid cloud model (OpenAI) for a free local model (Ollama) by changing **one line**
- What an LLM call actually returns (the `AIMessage` object: text, tokens, metadata)
- How to see every call in **LangSmith** without changing any code

## Files in this section

| File | What it shows | Needs |
|---|---|---|
| `01_hello_world_env_setup.py` | Loading keys from `.env` safely (prints a masked key) | nothing, or `OPENAI_API_KEY` |
| `02_summary_chain_openai.py` | Your first chain: `PromptTemplate \| ChatOpenAI` | `OPENAI_API_KEY` |
| `03_summary_chain_ollama.py` | The same chain on a local model, one-line change | Ollama + `gemma3:270m` |
| `04_inspect_the_response_object.py` | Every field of the returned `AIMessage` | `OPENAI_API_KEY` (or `--local`) |
| `sample_data.py` | The biography text that every script works on | — |

## Setup (one time)

```bash
cd section-01-langchain-foundations
uv sync                      # creates .venv and installs the exact locked versions
cp .env.example .env         # then paste your OPENAI_API_KEY into .env
```

For the local model (script 03 and `04 --local`):

1. Install Ollama from https://ollama.com/download and start it.
2. Download the tiny model once: `ollama pull gemma3:270m` (about 300 MB).
   Optional, much better answers: `ollama pull gemma4` (about 10 GB) and set `OLLAMA_MODEL=gemma4` in `.env`.

## Run

```bash
uv run python 01_hello_world_env_setup.py
uv run python 02_summary_chain_openai.py
uv run python 03_summary_chain_ollama.py
uv run python 04_inspect_the_response_object.py           # OpenAI
uv run python 04_inspect_the_response_object.py --local   # Ollama
```

## Expected output (script 02, sample)

```
Hello from Future with Shivank — AI Harness!
Model: gpt-5.6-sol

1. Headline: From newspaper boy in Rameswaram to India's "Missile Man" and President.

2. Summary for a 12-year-old:
- He was a scientist who helped India build its own rockets and put a satellite into space.
- People called him the "Missile Man of India", and later he became the country's 11th President.
- He loved teaching students and was giving a lecture when he died in 2015.

3. Interview question:
What did selling newspapers as a boy teach you that helped you as a scientist?
```

LLM wording changes on every run. The **structure** (headline, three bullets, one interview question) should stay the same, because the prompt spells out exactly those three parts. Every script has a detailed header that shows its full expected output.

## How it works

```
{"information": "<bio text>"}
          │
          ▼
  PromptTemplate      fills {information} into the template text
          │
          ▼
  ChatOpenAI / ChatOllama    sends the prompt to the model
          │
          ▼
  AIMessage           .text = the answer, .usage_metadata = tokens, ...
```

- `load_dotenv()` copies each `KEY=value` line from `.env` into the environment. Every SDK reads its key from there.
- `prompt | llm` builds a **RunnableSequence**. `chain.invoke({...})` runs it once.
- We print `response.text` instead of `response.content`. Newer OpenAI models reply through the Responses API, where `.content` can be a *list* of content blocks. `.text` always gives you a plain string.

## Optional: LangSmith tracing (no code changes)

Add these lines to `.env` and run any script again:

```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=section-01-langchain-foundations
```

Open https://smith.langchain.com. Each run appears as a `RunnableSequence` trace with two child steps:

- `PromptTemplate`: the filled prompt.
- `ChatOpenAI` / `ChatOllama`: latency, tokens and cost.

Compare a cloud run with a local run side by side.

## Formatting your code (optional)

```bash
uv run black .
uv run isort .
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `OPENAI_API_KEY is missing` | You haven't created `.env`. Run `cp .env.example .env` and paste your key. |
| `AuthenticationError` / 401 | The key is wrong or revoked. Create a new one at https://platform.openai.com/api-keys |
| `model ... does not exist` | Your account can't use the default model. Set `OPENAI_MODEL=` in `.env` to one it can use. |
| Ollama `ConnectionError` | The Ollama app isn't running. Start it, or run `ollama serve`. |
| `model 'gemma3:270m' not found` | Run `ollama pull gemma3:270m`. |

**Next:** [Section 2 — Your First Agent](../section-02-first-agent/README.md)
