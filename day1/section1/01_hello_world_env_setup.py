"""
===============================================================================
 Future with Shivank — AI Harness
 Section 1 · Script 01 · Hello world + loading API keys from a .env file
===============================================================================

WHAT THIS FILE TEACHES
----------------------
Before we talk to any LLM we need two boring-but-critical things working:

  1. Python can run our project (uv created a virtual environment for us).
  2. Our secret API keys are loaded from a `.env` file — NOT typed into code.

Why a .env file?
  • Keys never get committed to git (.env is listed in .gitignore).
  • Every library (OpenAI, LangSmith, Tavily, ...) automatically reads its key
    from an environment variable with a fixed name, e.g. OPENAI_API_KEY.
  • `load_dotenv()` copies each KEY=value line from .env into os.environ.

PREREQUISITES
-------------
    cp .env.example .env        # then paste your real OPENAI_API_KEY into .env

HOW TO RUN
----------
    uv run python 01_hello_world_env_setup.py

EXPECTED OUTPUT (with a key in .env)
------------------------------------
    Hello from Future with Shivank — AI Harness!
    OPENAI_API_KEY loaded ✅  (starts with 'sk-proj-', 164 characters)
    LANGSMITH_TRACING = (not set) → tracing is optional, see Script 02

EXPECTED OUTPUT (without a .env file)
-------------------------------------
    Hello from Future with Shivank — AI Harness!
    OPENAI_API_KEY is missing ❌  → copy .env.example to .env and add your key
    LANGSMITH_TRACING = (not set) → tracing is optional, see Script 02
===============================================================================
"""

import os

from dotenv import load_dotenv

# Reads ".env" from the current folder (and its parents) and puts every
# KEY=value pair into os.environ. Existing environment variables win, so a key
# exported in your terminal overrides the one in .env.
load_dotenv()


# -----------------------------------------------------------------------------
# describe_secret()
# -----------------------------------------------------------------------------
# Safely reports whether a secret is loaded WITHOUT printing it.
# Never print a full API key: terminal output ends up in screenshots, screen
# recordings and logs.
#
#   Input : name (str) — the environment variable to check, e.g. "OPENAI_API_KEY"
#   Output: (str) a human-friendly status line
def describe_secret(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        return f"{name} is missing ❌  → copy .env.example to .env and add your key"
    # Show only the first 8 characters, which identify the key TYPE
    # (e.g. 'sk-proj-' for an OpenAI project key) but are useless to an attacker.
    return f"{name} loaded ✅  (starts with {value[:8]!r}, {len(value)} characters)"


# ↑ Examples:
#   describe_secret("OPENAI_API_KEY") → "OPENAI_API_KEY loaded ✅  (starts with 'sk-proj-', 164 characters)"
#   describe_secret("NOT_A_REAL_VAR") → "NOT_A_REAL_VAR is missing ❌  → copy .env.example ..."


def main() -> None:
    print("Hello from Future with Shivank — AI Harness!")

    # Verify the key that scripts 02 and 04 need.
    print(describe_secret("OPENAI_API_KEY"))

    # LangSmith tracing is optional. When LANGSMITH_TRACING=true and
    # LANGSMITH_API_KEY are set, every LangChain call is recorded automatically
    # (no code changes needed) and you can inspect it at https://smith.langchain.com
    tracing = os.environ.get("LANGSMITH_TRACING", "(not set)")
    hint = (
        "→ tracing is optional, see Script 02"
        if tracing != "true"
        else "→ runs will appear in LangSmith"
    )
    print(f"LANGSMITH_TRACING = {tracing} {hint}")


if __name__ == "__main__":
    main()
