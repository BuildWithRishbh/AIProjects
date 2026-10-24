"""
===============================================================================
 Future with Shivank — AI Harness
 Section 2 · Script 04 · Structured output: make the agent return a Python object
===============================================================================

WHAT THIS FILE TEACHES
----------------------
Free-text answers are great for humans but painful for programs ("which part
of this paragraph is the link?"). With `response_format=<Pydantic model>` the
agent must ALSO return its answer as data that matches a schema we define:

    class AgentAnswer(BaseModel):
        answer: str
        sources: list[Source]        # Source = {title: str, url: str}

    result["structured_response"]
      →  AgentAnswer(answer="...", sources=[Source(title="...", url="https://...")])

Now your code can do `for s in result["structured_response"].sources: ...`
— e.g. show each meetup link as a button in an app — with no parsing at all.

HOW LANGCHAIN ENFORCES THE SCHEMA (two strategies, chosen automatically)
------------------------------------------------------------------------
• ProviderStrategy — the model provider supports native structured output
  (OpenAI, Anthropic, xAI ...). LangChain passes the schema to the API and the
  provider guarantees the JSON matches it. Used here with OpenAI.
• ToolStrategy — for models without native support. LangChain gives the LLM
  one extra "tool" whose arguments ARE the schema and asks it to call that
  tool to finish. Same result, different mechanism.
You can force one: response_format=ToolStrategy(AgentAnswer) or
ProviderStrategy(AgentAnswer) (import from langchain.agents.structured_output).

PREREQUISITES (.env)
--------------------
    OPENAI_API_KEY=sk-...
    TAVILY_API_KEY=tvly-...

HOW TO RUN
----------
    uv run python 04_agent_with_structured_output.py

EXPECTED OUTPUT (sample — live results change)
----------------------------------------------
    Hello from Future with Shivank — AI Harness!
    ... (message trail as in Script 03) ...

    ============================================================
    STRUCTURED RESPONSE
    ============================================================
    type: AgentAnswer
    answer:
      Here are 3 upcoming meetups in Bengaluru:
      1) <meetup name> — <date> — <venue>
      2) ...
    sources (3):
      - <meetup 1 title>
        https://www.meetup.com/...
      - <meetup 2 title>
        https://...
      - <meetup 3 title>
        https://...

NOTE ON THE LAST MESSAGE
------------------------
With ProviderStrategy (OpenAI) the last AIMessage contains the answer as JSON
text. With ToolStrategy the last message is a ToolMessage that says
"Returning structured response: ...". Either way, don't parse it yourself —
read result["structured_response"].

OPTIONAL VARIATION
------------------
Structured output works with ANY tool. To try it with the hand-written tool
from Script 02, copy its `tavily = TavilyClient()` line and `search` function
into this file and set `tools = [search]`.
===============================================================================
"""

import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

from langchain.agents import create_agent  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from langchain_tavily import TavilySearch  # noqa: E402

from agent_trace_printer import print_agent_trace  # noqa: E402

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-sol")

QUESTION = (
    "Find 3 upcoming AI or developer meetups in Bengaluru and give the date, "
    "venue and a link for each."
)


# -----------------------------------------------------------------------------
# The OUTPUT SCHEMA (Pydantic models)
# -----------------------------------------------------------------------------
# Pydantic classes describe the shape of the data we want back. The class
# docstrings and Field descriptions are sent to the LLM, so — just like tool
# docstrings — they are instructions for the model.


class Source(BaseModel):
    """A web page the answer is based on."""

    title: str = Field(description="Short title of the page, e.g. the event name")
    url: str = Field(description="Full link to the page, starting with https://")


# ↑ Example:  Source(title="Bengaluru AI Builders — October meetup",
#                    url="https://www.meetup.com/...")


class AgentAnswer(BaseModel):
    """The agent's final reply: a readable answer plus the pages it used."""

    answer: str = Field(description="The answer to the user's question, in plain text")
    sources: list[Source] = Field(
        default_factory=list,  # an empty list if the agent cites nothing
        description="Every web page used to write the answer",
    )


# ↑ Example:
#   AgentAnswer(answer="Here are 3 meetups ...",
#               sources=[Source(title="...", url="https://..."), ...])
#   .model_dump() → {"answer": "Here are 3 meetups ...",
#                    "sources": [{"title": "...", "url": "https://..."}, ...]}


llm = ChatOpenAI(model=OPENAI_MODEL)
search_tool = TavilySearch()
tools = [search_tool]

# response_format=AgentAnswer is the ONLY change compared with Script 03.
agent = create_agent(model=llm, tools=tools, response_format=AgentAnswer)


def main() -> None:
    print("Hello from Future with Shivank — AI Harness!")
    result = agent.invoke({"messages": [HumanMessage(content=QUESTION)]})

    # The usual message trail is still there ...
    print_agent_trace(result)

    # ... plus a NEW key holding a validated Pydantic object.
    structured: AgentAnswer = result["structured_response"]
    print("\n" + "=" * 60)
    print("STRUCTURED RESPONSE")
    print("=" * 60)
    print(f"type: {type(structured).__name__}")
    print("answer:")
    for line in structured.answer.splitlines():
        print(f"  {line}")
    print(f"sources ({len(structured.sources)}):")
    for source in structured.sources:
        print(f"  - {source.title}")
        print(f"    {source.url}")


if __name__ == "__main__":
    main()
