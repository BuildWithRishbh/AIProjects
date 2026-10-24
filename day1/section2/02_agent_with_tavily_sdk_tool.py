"""
===============================================================================
 Future with Shivank — AI Harness
 Section 2 · Script 02 · Give the agent REAL web search (your own Tavily tool)
===============================================================================

WHAT THIS FILE TEACHES
----------------------
Same agent as Script 01. The ONLY change is the inside of the `search` tool:
instead of a hard-coded string it now calls Tavily, a search API built for
LLMs (free tier: 1,000 credits/month at https://app.tavily.com).

    Script 01:  return "Bengaluru right now: 24°C, light drizzle, ..."
    Script 02:  return tavily.search(query=query)      ← real internet results

This is "writing your own tool around a vendor SDK". In Script 03 we replace
our hand-written tool with the vendor's ready-made LangChain tool and compare.

PREREQUISITES (.env)
--------------------
    OPENAI_API_KEY=sk-...
    TAVILY_API_KEY=tvly-...      ← must be EXACTLY this name; the SDK looks it up

HOW TO RUN
----------
    # default question (meetups — the agent may search several times)
    uv run python 02_agent_with_tavily_sdk_tool.py

    # any question you like, e.g. the weather question from Script 01
    uv run python 02_agent_with_tavily_sdk_tool.py "What's the weather like in Bengaluru right now?"

EXPECTED OUTPUT (sample — live web results change every day)
------------------------------------------------------------
    Hello from Future with Shivank — AI Harness!
    Question: Find 3 upcoming AI or developer meetups in Bengaluru and give the date, venue and a link for each.
    🔎 search tool called with: upcoming AI meetups Bengaluru
    🔎 search tool called with: developer meetups Bangalore this month
    ...                                   ← the LLM may call the tool several
                                            times IN PARALLEL with different queries

    The agent produced 6 messages:
      [0] HumanMessage : Find 3 upcoming AI or developer meetups in Bengaluru ...
      [1] AIMessage    : 🛠  tool call → search({'query': 'upcoming AI meetups Bengaluru'})
      [1] AIMessage    : 🛠  tool call → search({'query': 'developer meetups Bangalore this month'})
      [2] ToolMessage  : (search) {"query": "upcoming AI meetups Bengaluru", "results": [{"url": "https://www.meetup.com/...
      [3] ToolMessage  : (search) {"query": "developer meetups Bangalore this month", "results": [...
      ...
    FINAL ANSWER:
    Here are 3 upcoming meetups in Bengaluru:
    1) <meetup name> — <date> — <venue> — https://www.meetup.com/...
    2) ...

    For the weather question you get a live report instead, e.g.
    "It's 23°C and partly cloudy in Bengaluru right now (source: ...)".
===============================================================================
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent  # noqa: E402
from langchain.tools import tool  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from tavily import TavilyClient  # noqa: E402

from agent_trace_printer import print_agent_trace  # noqa: E402

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-sol")

DEFAULT_QUESTION = (
    "Find 3 upcoming AI or developer meetups in Bengaluru and give the date, "
    "venue and a link for each."
)

# The raw Tavily SDK client. It reads TAVILY_API_KEY from the environment.
tavily = TavilyClient()


# -----------------------------------------------------------------------------
# search()  —  our own tool, now backed by real web search
# -----------------------------------------------------------------------------
# Same name, same docstring, same argument as in Script 01, so the LLM sees the
# same "menu card". Only the implementation changed.
#
#   Input : query (str) e.g. "upcoming AI meetups Bengaluru"
#   Output: (dict) Tavily's JSON answer, roughly:
#           {"query": "...",
#            "results": [{"title": "...", "url": "https://...",
#                         "content": "<snippet>", "score": 0.93}, ...],
#            "response_time": 1.2, ...}
#   (The return type is a dict, not a str — LangChain turns it into JSON text
#    for the LLM automatically.)
@tool
def search(query: str) -> dict:
    """Look up live information on the internet.

    Use this for anything that changes over time and that you cannot know
    from memory: today's weather, recent news, current prices, upcoming events.

    Args:
        query: What to look up, as a short search phrase.

    Returns:
        The most relevant results, each with a title, URL and short snippet.
    """
    print(f"🔎 search tool called with: {query}")
    return tavily.search(query=query)  # tip: tavily.search(query=query, max_results=5)


# ↑ Example:
#   search.invoke({"query": "Bengaluru weather now"})
#   → {"query": "Bengaluru weather now",
#      "results": [{"title": "Weather in Bengaluru", "url": "https://www.weatherapi.com/", ...}, ...]}


llm = ChatOpenAI(model=OPENAI_MODEL)
tools = [search]
agent = create_agent(model=llm, tools=tools)


def main() -> None:
    print("Hello from Future with Shivank — AI Harness!")

    # sys.argv[1:] are the words typed after the script name, if any.
    question = " ".join(sys.argv[1:]) or DEFAULT_QUESTION
    print(f"Question: {question}")

    result = agent.invoke({"messages": [HumanMessage(content=question)]})
    print_agent_trace(result)


if __name__ == "__main__":
    main()
