"""Tests for scripts/compare_parsers.py (runs offline: in tests the NER and
LLM parsers fall back to rules, so all three must agree)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from compare_parsers import PARSERS, compare, fields  # noqa: E402
from resume_parser import parse_resume  # noqa: E402

RESUME = os.path.join(ROOT, "tests", "fixtures", "resume_title_case.docx")


def test_compares_all_three_parsers():
    assert [name for name, _ in PARSERS] == ["rules", "ner", "llm"]


def test_fields_view():
    view = fields(parse_resume(RESUME))
    assert list(view) == ["name", "email", "phone", "skills", "education", "jobs", "years", "projects"]
    assert view["skills"] == "Excel, Python, SQL, Tableau, pandas"          # sorted
    assert "Brightline Analytics | Junior Data Analyst | 24 mo" in view["jobs"]


def test_compare_counts_agreement(capsys):
    agreement = {pair: [0, 0] for pair in [("rules", "ner"), ("rules", "llm"), ("ner", "llm")]}
    compare(RESUME, agreement)
    assert all(counts == [8, 8] for counts in agreement.values())
    assert "all three parsers agree" in capsys.readouterr().out
