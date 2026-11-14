"""Shared helper for calling an LLM. Every agent uses ask_llm().

Each agent picks its own provider + model in a config dict at the top of
its file, e.g.

    LLM_CONFIG = {"provider": "gemini", "model": "gemini-3.5-flash"}

so switching models is a one-line change.

Providers:
    "gemini"  Google Gemini   key: GEMINI_API_KEY
    "groq"    Groq            key: GROQ_API_KEY

Keys are read from the .env file in the project root (see .env.example).
Keys are never printed or put in error messages.
"""
import json
import os
import re
import time

import requests
from dotenv import load_dotenv
from pydantic import ValidationError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"))

TIMEOUT_SECONDS = 60
RETRIES = 1               # extra tries after a "busy" error
RETRY_WAIT_SECONDS = 3

PROVIDERS = {
    "gemini": {
        "key_name": "GEMINI_API_KEY",
        "url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
    },
    "groq": {
        "key_name": "GROQ_API_KEY",
        "url": "https://api.groq.com/openai/v1/chat/completions",
    },
}


class LLMError(Exception):
    """Raised for any LLM problem: missing key, network/API error, bad JSON,
    or output that doesn't match the schema. Callers catch this one error
    to fall back to a non-LLM method."""


def get_api_key(provider):
    key_name = PROVIDERS[provider]["key_name"]
    key = os.environ.get(key_name)
    if not key:
        raise LLMError(f"{key_name} is not set (add it to .env)")
    return key


def _post(url, headers, body):
    # The only function that touches the network. Tests replace it with a fake.
    # "Busy" errors (429 rate limit, 5xx overloaded) are usually temporary,
    # so wait a moment and try again before giving up.
    for attempt in range(1 + RETRIES):
        try:
            response = requests.post(url, headers=headers, json=body, timeout=TIMEOUT_SECONDS)
        except requests.RequestException as e:
            raise LLMError(f"network error: {type(e).__name__}") from None
        if response.status_code == 200:
            return response.json()
        busy = response.status_code == 429 or response.status_code >= 500
        if not busy or attempt == RETRIES:
            raise LLMError(f"API returned HTTP {response.status_code}: {response.text[:200]}")
        time.sleep(RETRY_WAIT_SECONDS)


def _call_gemini(prompt, model, want_json):
    url = PROVIDERS["gemini"]["url"].format(model=model)
    headers = {"x-goog-api-key": get_api_key("gemini")}
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0}}
    if want_json:
        body["generationConfig"]["responseMimeType"] = "application/json"
    data = _post(url, headers, body)
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        raise LLMError("unexpected Gemini response shape") from None


def _call_groq(prompt, model, want_json):
    url = PROVIDERS["groq"]["url"]
    headers = {"Authorization": f"Bearer {get_api_key('groq')}"}
    body = {"model": model, "temperature": 0,
            "messages": [{"role": "user", "content": prompt}]}
    if want_json:
        body["response_format"] = {"type": "json_object"}
    data = _post(url, headers, body)
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise LLMError("unexpected Groq response shape") from None


def parse_json(text):
    # Models sometimes wrap JSON in ```json ... ``` fences; strip them first
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise LLMError("LLM did not return valid JSON") from None


def ask_llm(prompt, provider, model, schema=None):
    """Sends a prompt to the chosen provider/model.

    schema=None  -> returns the reply as plain text.
    schema=<pydantic model class> -> asks for JSON, checks it against the
        schema, and returns it as a dict. Raises LLMError if it doesn't fit.
    """
    if provider not in PROVIDERS:
        raise LLMError(f"unknown provider '{provider}' (use one of {sorted(PROVIDERS)})")
    call = _call_gemini if provider == "gemini" else _call_groq
    text = call(prompt, model, want_json=schema is not None)
    if schema is None:
        return text
    data = parse_json(text)
    try:
        return schema.model_validate(data).model_dump()
    except ValidationError as e:
        raise LLMError(f"LLM JSON did not match schema: {e.error_count()} error(s)") from None
