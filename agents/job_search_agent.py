"""Agent 2: AI job-search agent (Web Search & Opportunity Agent)."""
from typing import List, Literal, Optional

from pydantic import BaseModel

# One-line change to switch models. Must differ from Agent 1 (Gemini
# gemini-3.5-flash) and Agent 3 (Groq qwen/qwen3.8-27b).
LLM_CONFIG = {"provider": "groq", "model": "llama-3.3-70b-versatile"}

MAX_STEPS = 4
MAX_KEYWORDS = 5
LEVELS = ("entry", "mid", "senior")


class QueryModel(BaseModel):
    keywords: List[str] = []
    title: str = ""
    location: str = ""
    experience_level: str = ""


class Decision(BaseModel):
    thought: str = ""
    action: Literal["search", "finish"]
    query: Optional[QueryModel] = None