# Section 2 — Your First Agent

> **Future with Shivank — AI Harness** · LLM + tools = agent · real web search · structured output

## What you'll learn

- Why LLMs need **tools**, and what makes an **agent** (an LLM that decides which tool to call, in a loop)
- How to turn any Python function into a tool with `@tool`, and why the **docstring is written for the model**
- How to build an agent in one line with `create_agent(model, tools)`
- How to read an agent run message by message (Human → AI tool call → Tool result → AI answer)
- Your own tool around a vendor SDK vs the vendor's ready-made LangChain tool
- **Structured output**: making the agent return a validated Pydantic object

## Files in this section

| File | What it shows | Needs |
|---|---|---|
| `01_agent_with_placeholder_tool.py` | The agent loop with a fake `search` tool that always returns the same Bengaluru weather | `OPENAI_API_KEY` |
| `02_agent_with_tavily_sdk_tool.py` | Your own `search` tool that calls the Tavily SDK (real web search) | + `TAVILY_API_KEY` |
| `03_agent_with_builtin_tavily_tool.py` | The vendor's ready-made `TavilySearch` tool | + `TAVILY_API_KEY` |
| `04_agent_with_structured_output.py` | `response_format=AgentAnswer` → `result["structured_response"]` with titled sources | + `TAVILY_API_KEY` |
| `agent_trace_printer.py` | Helper that prints the agent's messages one per line | — |

Each script changes **one thing** compared with the previous one. Open two files side by side to see the difference.

## Setup (one time)

```bash
cd section-02-first-agent
uv sync
cp .env.example .env      # add OPENAI_API_KEY and TAVILY_API_KEY
```

Get a free Tavily key (1,000 credits per month) at https://app.tavily.com. The variable must be named exactly `TAVILY_API_KEY`.

## Run

```bash
uv run python 01_agent_with_placeholder_tool.py
uv run python 02_agent_with_tavily_sdk_tool.py        # default: Bengaluru tech meetups
uv run python 02_agent_with_tavily_sdk_tool.py "What's the weather like in Bengaluru right now?"
uv run python 03_agent_with_builtin_tavily_tool.py
uv run python 04_agent_with_structured_output.py
```

## Expected output (script 01)

```
Hello from Future with Shivank — AI Harness!
🔎 search tool called with: Bengaluru weather now

The agent produced 4 messages:
  [0] HumanMessage : What's the weather like in Bengaluru right now?
  [1] AIMessage    : 🛠  tool call → search({'query': 'Bengaluru weather now'})
  [2] ToolMessage  : (search) Bengaluru right now: 24°C, light drizzle, humidity 78%
  [3] AIMessage    : It's about 24°C in Bengaluru right now, with light drizzle and 78% humidity — keep an umbrella handy!

FINAL ANSWER:
It's about 24°C in Bengaluru right now, with light drizzle and 78% humidity — keep an umbrella handy!
```

The query the model writes and its final wording can vary slightly. Scripts 02–04 use live web results (by default: *"Find 3 upcoming AI or developer meetups in Bengaluru and give the date, venue and a link for each."*), so their answers change from day to day. Each file's header shows a sample.

## How it works

```
question ──► LLM ──"call search('Bengaluru weather now')"──► LangChain runs search()
               ▲                                                     │
               └──── ToolMessage("Bengaluru right now: 24°C, ...") ───┘
LLM ──► "It's about 24°C in Bengaluru right now ..."   (no more tool calls → done)
```

- The LLM **never runs code**. It only *requests* a tool call, and LangChain runs the function.
- `@tool` sends the function's **name, type hints and docstring** to the model. That's all the model knows about the tool.
- With `response_format=...`, LangChain picks one of two strategies:
  - **ProviderStrategy**: native structured output, used by OpenAI, Anthropic and others.
  - **ToolStrategy**: an extra "answer" tool, for models without native support.

  You can force either one with `ToolStrategy(...)` or `ProviderStrategy(...)`.

## Optional: LangSmith tracing (highly recommended here)

Add `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY=...` and `LANGSMITH_PROJECT=section-02-first-agent` to `.env`. Each run shows the whole loop:

1. The first `ChatOpenAI` call returns a tool call.
2. The tool runs.
3. The second `ChatOpenAI` call returns the final answer.

## Troubleshooting

| Problem | Fix |
|---|---|
| `MissingAPIKeyError` / Tavily 401 | `TAVILY_API_KEY` is missing or misspelled in `.env`. |
| The agent answers without searching | Make the tool docstring more explicit, or ask something that clearly needs fresh data (weather, news, upcoming events). |
| Meetup results look old or irrelevant | Live search quality varies. Add a month to the question, e.g. "... in Bengaluru in November", and run it again. |
| Very long output | That's the raw search JSON. The trace printer shortens each message to one line. |
| `result["structured_response"]` KeyError | You're running script 03; only script 04 sets `response_format`. |

**Previous:** [Section 1](../section-01-langchain-foundations/README.md) · **Next:** [Section 3 — The Agent Loop with LangChain](../section-03-agent-loop-langchain/README.md)
