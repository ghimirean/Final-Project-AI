"""Tests for the NER training/evaluation code (scripts/train_ner.py)."""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "tests", "fixtures", "ner"))

from make_ner_data import check_alignment  # noqa: E402
from train_ner import LABELS, MODEL_DIR, evaluate, load_records, split, to_examples, train  # noqa: E402

RECORDS = load_records("resumes.jsonl")
HANDMADE = load_records("handmade_eval.jsonl")


# ---------- split ----------

def test_split_is_80_20_disjoint_and_fixed():
    train_set, test_set = split(RECORDS)
    assert (len(train_set), len(test_set)) == (48, 12)
    assert not {r["id"] for r in train_set} & {r["id"] for r in test_set}
    assert split(RECORDS) == (train_set, test_set)            # same seed, same split


# ---------- evaluate (with a fake model) ----------

class FakeEnt:
    def __init__(self, start, end, label):
        self.start_char, self.end_char, self.label_ = start, end, label


class FakeDoc:
    def __init__(self, ents):
        self.ents = ents


def fake_nlp(predictions):
    return lambda text: FakeDoc([FakeEnt(*p) for p in predictions])


RECORD = {"id": "r", "text": "Ada Lovelace Python",
          "entities": [[0, 12, "PERSON", "Ada Lovelace"], [13, 19, "SKILL", "Python"]]}


def test_evaluate_perfect():
    scores = evaluate(fake_nlp([(0, 12, "PERSON"), (13, 19, "SKILL")]), [RECORD])
    assert scores["overall"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 2}


def test_evaluate_exact_match_only():
    # Wrong label on Python, and "Ada" alone is only part of the name
    scores = evaluate(fake_nlp([(0, 3, "PERSON"), (13, 19, "ORG")]), [RECORD])
    assert scores["per_label"]["PERSON"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 1}
    assert scores["per_label"]["SKILL"]["recall"] == 0.0
    assert scores["per_label"]["ORG"]["precision"] == 0.0


def test_evaluate_precision_and_recall():
    # One correct, one extra wrong SKILL, PERSON missed: P = 1/2, R = 1/2
    scores = evaluate(fake_nlp([(13, 19, "SKILL"), (4, 12, "SKILL")]), [RECORD])
    assert scores["overall"]["precision"] == 0.5
    assert scores["overall"]["recall"] == 0.5


def test_evaluate_ignores_other_labels():
    scores = evaluate(fake_nlp([(0, 12, "PERSON"), (13, 19, "SKILL"), (0, 3, "GPE")]), [RECORD])
    assert scores["overall"]["precision"] == 1.0


# ---------- data used for training/evaluation ----------

def test_handmade_annotations_align():
    assert len(HANDMADE) == 3
    assert check_alignment(HANDMADE) == []


def test_examples_carry_the_annotations():
    import spacy
    nlp = spacy.blank("en")
    example = to_examples(nlp, RECORDS[:1])[0]
    gold = [(e.start_char, e.end_char, e.label_) for e in example.reference.ents]
    assert gold == [(s, e, label) for s, e, label, _ in RECORDS[0]["entities"]]


def test_training_runs_and_adds_new_labels():
    # Tiny smoke run: 1 epoch on 2 resumes. Checks the training code works,
    # not the quality of the result.
    nlp = train(RECORDS[:2], epochs=1, log=lambda *a: None)
    assert {"DEGREE", "JOB_TITLE", "SKILL"} <= set(nlp.get_pipe("ner").labels)


# ---------- the trained model (only if it has been built locally) ----------

@pytest.mark.skipif(not os.path.exists(os.path.join(MODEL_DIR, "metrics.json")),
                    reason="models/resume_ner not built; run python scripts/train_ner.py")
def test_saved_model_metrics():
    with open(os.path.join(MODEL_DIR, "metrics.json"), encoding="utf-8") as f:
        metrics = json.load(f)
    assert metrics["fine_tuned"]["synthetic_test"]["overall"]["f1"] >= 0.9
    assert metrics["fine_tuned"]["handmade"]["overall"]["f1"] >= 0.8
    assert (metrics["fine_tuned"]["synthetic_test"]["overall"]["f1"]
            > metrics["baseline"]["synthetic_test"]["overall"]["f1"])
    assert set(metrics["fine_tuned"]["synthetic_test"]["per_label"]) == set(LABELS)
