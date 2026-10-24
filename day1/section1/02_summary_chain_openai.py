"""
===============================================================================
 Future with Shivank — AI Harness
 Section 1 · Script 02 · Your first LangChain chain: PromptTemplate | ChatOpenAI
===============================================================================

WHAT THIS FILE TEACHES
----------------------
The three building blocks you will use in EVERY later section:

  1. PromptTemplate  – a prompt with {placeholders} that you fill at run time.
  2. Chat model      – a Python object that talks to an LLM (here: OpenAI).
  3. Chain (`|`)     – the pipe operator connects blocks, so the output of one
                       becomes the input of the next (LCEL = LangChain
                       Expression Language).

        {"information": "<bio text>"}
                  │
                  ▼
        ┌───────────────────┐   fills {information} in the template
        │  PromptTemplate   │
        └─────────┬─────────┘
                  │  StringPromptValue("Here is a short biography: ... interview")
                  ▼
        ┌───────────────────┐   sends the prompt to OpenAI
        │    ChatOpenAI     │
        └─────────┬─────────┘
                  │  AIMessage(content=..., response_metadata=...)
                  ▼
           response.text  → the answer as a plain string

PREREQUISITES (.env)
--------------------
    OPENAI_API_KEY=sk-...
    OPENAI_MODEL=gpt-5.6-sol          # optional — this is the default
    # Optional tracing (see the bottom of this docstring):
    LANGSMITH_TRACING=true
    LANGSMITH_API_KEY=lsv2_...
    LANGSMITH_PROJECT=section-01-langchain-foundations

HOW TO RUN
----------
    uv run python 02_summary_chain_openai.py

EXPECTED OUTPUT (sample — the wording changes on every run)
-----------------------------------------------------------
    Hello from Future with Shivank — AI Harness!
    Model: gpt-5.6-sol

    1. Headline: From newspaper boy in Rameswaram to India's "Missile Man" and President.

    2. Summary for a 12-year-old:
    - He was a scientist who helped India build its own rockets and put a
      satellite into space.
    - People called him the "Missile Man of India", and later he became the
      country's 11th President.
    - He loved teaching students and was giving a lecture when he died in 2015.

    3. Interview question:
    What did selling newspapers as a boy teach you that helped you as a scientist?

LANGSMITH TRACING (optional, zero code changes)
-----------------------------------------------
Add the three LANGSMITH_* lines above to .env and run the script again. Open
https://smith.langchain.com → project "section-01-langchain-foundations" and
you will see one trace called "RunnableSequence" with two children:
"PromptTemplate" (the filled prompt) and "ChatOpenAI" (the model call, with
latency, token count and cost). LangChain sends traces automatically whenever
those environment variables are present.
===============================================================================
"""

import os

from dotenv import load_dotenv

# Load OPENAI_API_KEY (and optional LANGSMITH_* settings) from .env.
load_dotenv()

from langchain_core.prompts import PromptTemplate  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

from sample_data import PERSON_INFORMATION  # noqa: E402

# Which OpenAI model to use. Change it in .env (OPENAI_MODEL=...) — no code edit.
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-sol")

# -----------------------------------------------------------------------------
# The prompt TEMPLATE
# -----------------------------------------------------------------------------
# {information} is a placeholder. PromptTemplate replaces it with real text
# when the chain runs. Everything else is sent to the model as-is.
# The numbered list is an "output contract": it tells the model exactly which
# three parts to return, so every run has the same shape.
PROFILE_TEMPLATE = """Here is a short biography:
{information}

Using only this text, give me:
1. A one-line headline about this person
2. A three-bullet summary a 12-year-old would understand
3. One question you would ask them in an interview
"""


# -----------------------------------------------------------------------------
# build_profile_chain()
# -----------------------------------------------------------------------------
# Creates the chain  PromptTemplate | ChatOpenAI.
#
#   Input : model_name (str) e.g. "gpt-5.6-sol"
#   Output: a Runnable. Call it with chain.invoke({"information": "..."})
def build_profile_chain(model_name: str):
    # input_variables lists every {placeholder} the template expects.
    profile_prompt_template = PromptTemplate(
        input_variables=["information"], template=PROFILE_TEMPLATE
    )

    # temperature=0 asks for the most deterministic answer the model can give.
    # NOTE: GPT-5.x reasoning models only support their default temperature;
    # LangChain quietly drops the setting for them, so this line is safe for
    # every model (and it does matter for local models in Script 03).
    llm = ChatOpenAI(temperature=0, model=model_name)

    # The pipe operator builds a "RunnableSequence": prompt first, then model.
    return profile_prompt_template | llm


# ↑ Example:
#   chain = build_profile_chain("gpt-5.6-sol")
#   chain.invoke({"information": "Sachin Tendulkar is a former Indian cricketer ..."})
#   → AIMessage(content=..., response_metadata={"model_name": "gpt-5.6-sol", ...})


def main() -> None:
    print("Hello from Future with Shivank — AI Harness!")
    print(f"Model: {OPENAI_MODEL}\n")

    chain = build_profile_chain(OPENAI_MODEL)

    # invoke() runs the whole chain once. The dict keys must match the
    # template's placeholders ({information}).
    response = chain.invoke(input={"information": PERSON_INFORMATION})

    # `response` is an AIMessage (see Script 04 for everything inside it).
    # `.text` returns the answer as a plain string. We use `.text` instead of
    # `.content` because newer OpenAI models reply through the Responses API,
    # where `.content` may be a LIST of content blocks rather than a string.
    print(response.text)


if __name__ == "__main__":
    main()
