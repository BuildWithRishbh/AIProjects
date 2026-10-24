"""
===============================================================================
 Future with Shivank — AI Harness
 Section 1 · Script 03 · Same chain, LOCAL open-weights model (Ollama)
===============================================================================

WHAT THIS FILE TEACHES
----------------------
LangChain gives every chat model the same interface. To move from a paid cloud
model (OpenAI) to a free model running on YOUR laptop, you change exactly ONE
line — the model object. The prompt, the chain and invoke() stay identical:

    Script 02:  llm = ChatOpenAI(temperature=0, model="gpt-5.6-sol")
    Script 03:  llm = ChatOllama(temperature=0, model="gemma3:270m")   ← only change

Ollama is a free app that downloads open-weights models (Gemma, Qwen, Llama,
...) and serves them on http://localhost:11434. No API key, no cost, and your
data never leaves your machine.

PREREQUISITES
-------------
1. Install Ollama from https://ollama.com/download and start it.
2. Download the model once (≈300 MB for the tiny 270-million-parameter Gemma):
       ollama pull gemma3:270m
3. Optional — use a bigger/better local model by adding to .env:
       OLLAMA_MODEL=gemma4          # much better answers, ≈10 GB download

HOW TO RUN
----------
    uv run python 03_summary_chain_ollama.py

EXPECTED OUTPUT (sample — small local models vary a lot)
--------------------------------------------------------
    Hello from Future with Shivank — AI Harness!
    Model: gemma3:270m (local, via Ollama)

    **Headline:** Abdul Kalam: The Missile Man of India

    **Summary:**
    * Abdul Kalam was a scientist who worked at ISRO and DRDO.
    * He became the 11th President of India.
    * He wrote a book called "Wings of Fire".
    * He was born in Rameswaram ...

WHAT TO NOTICE
--------------
The local answer comes back FAST, but a 270M-parameter model is tiny compared
with GPT-class models. It often ignores parts of the instructions — for
example it may write four bullets instead of three, or forget the interview
question. That is the
classic trade-off: cost/privacy/speed versus quality. Try OLLAMA_MODEL=gemma4
and compare.

TROUBLESHOOTING
---------------
  ConnectionError / "Failed to connect to Ollama"  → the Ollama app is not running.
  "model 'gemma3:270m' not found"                  → run: ollama pull gemma3:270m
===============================================================================
"""

import os

from dotenv import load_dotenv

# Ollama needs no API key, but .env may still hold LANGSMITH_* tracing settings
# and an optional OLLAMA_MODEL override.
load_dotenv()

from langchain_core.prompts import PromptTemplate  # noqa: E402
from langchain_ollama import ChatOllama  # noqa: E402

from sample_data import PERSON_INFORMATION  # noqa: E402

# (Script 02 imports ChatOpenAI here instead — that is the only difference.)

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:270m")

# Exactly the same template as Script 02.
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
#   Input : model_name (str) — any model you have pulled with `ollama pull`
#   Output: a Runnable chain  PromptTemplate | ChatOllama
def build_profile_chain(model_name: str):
    profile_prompt_template = PromptTemplate(
        input_variables=["information"], template=PROFILE_TEMPLATE
    )

    # llm = ChatOpenAI(temperature=0, model="gpt-5.6-sol")   # ← Script 02's line
    llm = ChatOllama(temperature=0, model=model_name)  # ← talks to localhost:11434

    return profile_prompt_template | llm


# ↑ Example:
#   chain = build_profile_chain("gemma3:270m")
#   chain.invoke({"information": "..."}).text  → "**Headline:** ..."


def main() -> None:
    print("Hello from Future with Shivank — AI Harness!")
    print(f"Model: {OLLAMA_MODEL} (local, via Ollama)\n")

    chain = build_profile_chain(OLLAMA_MODEL)
    response = chain.invoke(input={"information": PERSON_INFORMATION})

    # For ChatOllama `.content` is already a string; `.text` works for every
    # provider, so we use it everywhere in the course.
    print(response.text)


if __name__ == "__main__":
    main()
