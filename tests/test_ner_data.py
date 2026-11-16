"""Checks the annotated NER training data (tests/fixtures/ner/resumes.jsonl)."""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NER_DIR = os.path.join(ROOT, "tests", "fixtures", "ner")
sys.path.insert(0, NER_DIR)

from make_ner_data import LABELS, check_alignment, generate  # noqa: E402

with open(os.path.join(NER_DIR, "resumes.jsonl"), encoding="utf-8") as f:
    RECORDS = [json.loads(line) for line in f]


def test_thirty_resumes_with_unique_ids():
    assert len(RECORDS) == 30
    assert len({r["id"] for r in RECORDS}) == 30


def test_file_matches_generator():
    # The committed file is exactly what the generator produces (seeded)
    assert generate() == RECORDS


def test_offsets_match_text():
    for r in RECORDS:
        for start, end, label, text in r["entities"]:
            assert r["text"][start:end] == text, (r["id"], text)
            assert label in LABELS


def test_spans_align_with_spacy_tokens():
    assert check_alignment(RECORDS) == []


def test_no_overlapping_entities():
    for r in RECORDS:
        spans = sorted((s, e) for s, e, _, _ in r["entities"])
        for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
            assert e1 <= s2, (r["id"], (s1, e1), (s2, e2))


def test_every_resume_has_core_entities():
    for r in RECORDS:
        labels = [e[2] for e in r["entities"]]
        assert labels.count("PERSON") == 1, r["id"]
        assert labels.count("DEGREE") == 1, r["id"]
        assert "JOB_TITLE" in labels and "SKILL" in labels and "DATE" in labels, r["id"]


def test_every_label_is_well_represented():
    counts = {label: sum(e[2] == label for r in RECORDS for e in r["entities"]) for label in LABELS}
    assert all(count >= 30 for count in counts.values()), counts


def test_data_is_fake():
    for r in RECORDS:
        emails = [w for w in r["text"].split() if "@" in w]
        assert all(w.endswith("@example.com") for w in emails), r["id"]


@pytest.mark.parametrize("label, not_labeled", [
    ("SKILL", "github.com"),          # links are not skills
    ("JOB_TITLE", "Team Lead"),       # project roles are not job titles
    ("DATE", "Expected"),             # "Expected" is not part of the date
])
def test_labeling_guidelines(label, not_labeled):
    for r in RECORDS:
        for _, _, entity_label, text in r["entities"]:
            if entity_label == label:
                assert not_labeled not in text, (r["id"], text)
