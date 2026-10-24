"""
===============================================================================
 Future with Shivank — AI Harness
 Section 2 · Helper: print an agent run as a readable, numbered message trail
===============================================================================

WHY THIS FILE EXISTS
--------------------
`agent.invoke(...)` returns a dict whose "messages" list is the agent's whole
conversation. `print(result)` dumps it as one giant unreadable line. This
helper prints the same information one message per line, so you can SEE the
ReAct loop:

    [0] HumanMessage : What's the weather like in Bengaluru right now?
    [1] AIMessage    : 🛠  tool call → search({'query': 'Bengaluru weather now'})
    [2] ToolMessage  : (search) Bengaluru right now: 24°C, light drizzle, humidity 78%
    [3] AIMessage    : It's about 24°C in Bengaluru right now, with light drizzle ...

  [0] the question  →  [1] the model DECIDES to call a tool  →
  [2] our code RUNS the tool and adds the result  →  [3] the model ANSWERS.

Every script in this section imports it:
    from agent_trace_printer import print_agent_trace
===============================================================================
"""


# -----------------------------------------------------------------------------
# _one_line()
# -----------------------------------------------------------------------------
# Squashes any text to a single line and cuts it to `limit` characters, so long
# search results don't flood the terminal.
#
#   _one_line("a\nb\nc", 10)   → "a b c"
#   _one_line("x" * 500, 10)   → "xxxxxxxxxx … (+490 chars)"
def _one_line(text: str, limit: int = 160) -> str:
    flat = " ".join(str(text).split())
    if len(flat) <= limit:
        return flat
    return f"{flat[:limit]} … (+{len(flat) - limit} chars)"


# -----------------------------------------------------------------------------
# print_agent_trace()
# -----------------------------------------------------------------------------
#   Input : result (dict) — the value returned by agent.invoke(...)
#           It always has a "messages" key; with response_format=... it also
#           has a "structured_response" key (see Script 04).
#   Output: None (prints to the terminal)
def print_agent_trace(result: dict) -> None:
    messages = result["messages"]
    print(f"\nThe agent produced {len(messages)} messages:")
    for index, message in enumerate(messages):
        kind = type(message).__name__
        # AIMessages that call tools have an empty text and a tool_calls list.
        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            for call in tool_calls:
                print(
                    f"  [{index}] {kind:<13}: 🛠  tool call → {call['name']}({_one_line(call['args'], 120)})"
                )
        elif kind == "ToolMessage":
            print(f"  [{index}] {kind:<13}: ({message.name}) {_one_line(message.text)}")
        else:
            print(f"  [{index}] {kind:<13}: {_one_line(message.text)}")

    # The final answer is simply the text of the LAST message.
    print("\nFINAL ANSWER:")
    print(messages[-1].text)


# ↑ Example:
#   result = agent.invoke({"messages": [HumanMessage(content="What's the weather like in Bengaluru right now?")]})
#   print_agent_trace(result)
#   → prints the 4-line trail shown in the module docstring, then the final answer.
