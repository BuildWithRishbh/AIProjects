"""
===============================================================================
 Future with Shivank — AI Harness
 Section 2 · Script 01 · Your first agent: an LLM + one (fake) tool
===============================================================================

WHAT THIS FILE TEACHES
----------------------
An LLM on its own can only produce text. It doesn't know today's weather, this
week's news or the current price of anything. An AGENT fixes that:

    agent = LLM (the "brain" that decides)  +  TOOLS (functions it may call)

LangChain's `create_agent(model, tools)` builds the whole loop for you:
  1. send the question + the list of tools to the LLM
  2. if the LLM answers "please call tool X with args Y" → run X(Y)
  3. send the tool result back to the LLM, repeat until it answers in plain text

To focus purely on the MECHANICS, our `search` tool is a placeholder: whatever
you ask, it returns the same fixed weather report for Bengaluru. In Script 02
we swap in a real web search — and nothing else changes.

PREREQUISITES (.env)
--------------------
    OPENAI_API_KEY=sk-...
    OPENAI_MODEL=gpt-5.6-sol            # optional (default)
    LANGSMITH_TRACING=true              # optional — see the trace in LangSmith
    LANGSMITH_API_KEY=lsv2_...
    LANGSMITH_PROJECT=section-02-first-agent

HOW TO RUN
----------
    uv run python 01_agent_with_placeholder_tool.py

EXPECTED OUTPUT (the query and final wording may vary slightly)
---------------------------------------------------------------
    Hello from Future with Shivank — AI Harness!
    🔎 search tool called with: Bengaluru weather now      ← printed INSIDE our tool

    The agent produced 4 messages:
      [0] HumanMessage : What's the weather like in Bengaluru right now?
      [1] AIMessage    : 🛠  tool call → search({'query': 'Bengaluru weather now'})
      [2] ToolMessage  : (search) Bengaluru right now: 24°C, light drizzle, humidity 78%
      [3] AIMessage    : It's about 24°C in Bengaluru right now, with light drizzle and 78% humidity — keep an umbrella handy!

    FINAL ANSWER:
    It's about 24°C in Bengaluru right now, with light drizzle and 78% humidity — keep an umbrella handy!

WHAT TO NOTICE
--------------
• The LLM wrote the query "Bengaluru weather now" itself — it read our
  docstring and decided the tool was worth calling.
• The LLM never runs code. Message [1] is only a REQUEST; LangChain executed
  the function and stored the result in message [2].
• In LangSmith (optional) you'll see two "ChatOpenAI" calls with the tool
  call in between — the agent loop made visible.
===============================================================================
"""

import os

from dotenv import load_dotenv

# Load keys BEFORE importing/creating anything that needs them.
load_dotenv()

from langchain.agents import create_agent  # noqa: E402
from langchain.tools import tool  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

from agent_trace_printer import print_agent_trace  # noqa: E402

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-sol")

QUESTION = "What's the weather like in Bengaluru right now?"


# -----------------------------------------------------------------------------
# search()  —  a LangChain TOOL
# -----------------------------------------------------------------------------
# @tool turns this ordinary function into something an agent can call.
# LangChain reads three things and sends them to the LLM as the tool's
# "menu card":
#   • the NAME        → "search"
#   • the TYPE HINTS  → query: str, returns str
#   • the DOCSTRING   → when to use the tool and what each argument means
# The LLM decides WHETHER and HOW to call the tool purely from that card, so
# the docstring is written for the MODEL: say clearly when the tool helps.
#
#   Input : query (str) — written by the LLM, e.g. "Bengaluru weather now"
#   Output: (str) the "search result" (hard-coded for now)
@tool
def search(query: str) -> str:
    """Look up live information on the internet.

    Use this for anything that changes over time and that you cannot know
    from memory: today's weather, recent news, current prices, upcoming events.

    Args:
        query: What to look up, as a short search phrase.

    Returns:
        The most relevant result as plain text.
    """
    print(f"🔎 search tool called with: {query}")
    return "Bengaluru right now: 24°C, light drizzle, humidity 78%"


# ↑ You can call a tool directly too (handy for testing):
#   search.invoke({"query": "anything at all"})
#   → prints   🔎 search tool called with: anything at all
#   → returns  "Bengaluru right now: 24°C, light drizzle, humidity 78%"
#   search.name        → "search"
#   search.description → "Look up live information on the internet. ..."


# -----------------------------------------------------------------------------
# Build the agent: a reasoning engine (LLM) + a list of tools
# -----------------------------------------------------------------------------
llm = ChatOpenAI(model=OPENAI_MODEL)
tools = [search]
agent = create_agent(model=llm, tools=tools)


def main() -> None:
    print("Hello from Future with Shivank — AI Harness!")

    # An agent's input is a dict with a "messages" list — the same shape as a
    # chat conversation. HumanMessage = "this was typed by the user".
    result = agent.invoke({"messages": [HumanMessage(content=QUESTION)]})

    # `result` is a dict: {"messages": [HumanMessage, AIMessage, ToolMessage, AIMessage]}
    # print(result)          # ← uncomment to see the raw dict
    print_agent_trace(result)


if __name__ == "__main__":
    main()
