"""Trains Agent 1's resume NER model and reports precision / recall / F1.

Starts from spaCy's pretrained English model (en_core_web_sm) and
fine-tunes its NER component on our 30 annotated fake resumes
(tests/fixtures/ner/resumes.jsonl) to recognize:
    PERSON, ORG, DEGREE, JOB_TITLE, SKILL, DATE
(en_core_web_sm already knows PERSON, ORG and DATE; DEGREE, JOB_TITLE and
SKILL are new labels it learns here.)

Data split (fixed seed, so every run uses the same split):
    24 resumes for training, 6 held out for testing (80/20).
Two test sets are reported:
    1. "synthetic test": the 6 held-out generated resumes
    2. "handmade": the 3 hand-written fixture resumes, whose layouts the
       generator never produces, a fairer check of generalization
Scores are exact-match: an entity counts as correct only if its start,
end AND label all match the annotation.

Run from project root (about a minute):
    python scripts/train_ner.py
Saves the model to models/resume_ner/ (git-ignored; re-run this script to
rebuild it) together with models/resume_ner/metrics.json.
"""
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NER_DIR = os.path.join(ROOT, "tests", "fixtures", "ner")
MODEL_DIR = os.path.join(ROOT, "models", "resume_ner")

LABELS = ["PERSON", "ORG", "DEGREE", "JOB_TITLE", "SKILL", "DATE"]
BASE_MODEL = "en_core_web_sm"
SEED = 0
TEST_SIZE = 6
EPOCHS = 30
DROPOUT = 0.3


def load_records(filename):
    with open(os.path.join(NER_DIR, filename), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def split(records, test_size=TEST_SIZE, seed=SEED):
    """Shuffles a copy with a fixed seed and returns (train, test)."""
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    return shuffled[test_size:], shuffled[:test_size]


def to_examples(nlp, records):
    from spacy.training import Example
    return [Example.from_dict(nlp.make_doc(r["text"]),
                              {"entities": [(s, e, label) for s, e, label, _ in r["entities"]]})
            for r in records]


def evaluate(nlp, records, labels=LABELS):
    """Exact-match precision, recall and F1 per label, plus overall (micro).
    Predictions with labels we don't use (e.g. GPE, CARDINAL) are ignored."""
    counts = {label: {"tp": 0, "fp": 0, "fn": 0} for label in labels}
    for r in records:
        gold = {(s, e, label) for s, e, label, _ in r["entities"] if label in labels}
        pred = {(ent.start_char, ent.end_char, ent.label_) for ent in nlp(r["text"]).ents
                if ent.label_ in labels}
        for s, e, label in pred & gold:
            counts[label]["tp"] += 1
        for s, e, label in pred - gold:
            counts[label]["fp"] += 1
        for s, e, label in gold - pred:
            counts[label]["fn"] += 1

    def prf(c):
        p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
        r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3),
                "support": c["tp"] + c["fn"]}

    total = {k: sum(c[k] for c in counts.values()) for k in ("tp", "fp", "fn")}
    return {"per_label": {label: prf(c) for label, c in counts.items()}, "overall": prf(total)}


def train(train_records, epochs=EPOCHS, seed=SEED, log=print):
    import spacy
    from spacy.util import minibatch
    spacy.util.fix_random_seed(seed)
    nlp = spacy.load(BASE_MODEL)
    ner = nlp.get_pipe("ner")
    for label in LABELS:
        ner.add_label(label)                       # adds DEGREE, JOB_TITLE, SKILL
    examples = to_examples(nlp, train_records)
    rng = random.Random(seed)
    with nlp.select_pipes(enable=["ner"]):         # only the NER part is trained
        optimizer = nlp.resume_training()
        for epoch in range(1, epochs + 1):
            rng.shuffle(examples)
            losses = {}
            for batch in minibatch(examples, size=4):
                nlp.update(batch, drop=DROPOUT, sgd=optimizer, losses=losses)
            if epoch == 1 or epoch % 5 == 0:
                log(f"  epoch {epoch:>2}  loss {losses['ner']:.1f}")
    return nlp


def print_table(title, scores):
    print(f"\n{title}")
    print(f"  {'label':<10}{'precision':>10}{'recall':>8}{'F1':>7}{'support':>9}")
    for label, s in list(scores["per_label"].items()) + [("OVERALL", scores["overall"])]:
        print(f"  {label:<10}{s['precision']:>10.3f}{s['recall']:>8.3f}{s['f1']:>7.3f}{s['support']:>9}")


if __name__ == "__main__":
    import spacy
    records = load_records("resumes.jsonl")
    handmade = load_records("handmade_eval.jsonl")
    train_records, test_records = split(records)
    print(f"train: {len(train_records)} resumes, test: {len(test_records)} resumes "
          f"({', '.join(r['id'] for r in test_records)}), handmade: {len(handmade)}")

    base = spacy.load(BASE_MODEL)
    baseline = {"synthetic_test": evaluate(base, test_records), "handmade": evaluate(base, handmade)}

    print(f"\ntraining {BASE_MODEL} NER for {EPOCHS} epochs ...")
    nlp = train(train_records)
    results = {"synthetic_test": evaluate(nlp, test_records), "handmade": evaluate(nlp, handmade)}

    print_table(f"BASELINE {BASE_MODEL} (no fine-tuning), synthetic test set", baseline["synthetic_test"])
    print_table("FINE-TUNED, synthetic test set (6 held-out generated resumes)", results["synthetic_test"])
    print_table("FINE-TUNED, handmade set (3 resumes in unseen layouts)", results["handmade"])

    os.makedirs(MODEL_DIR, exist_ok=True)
    nlp.to_disk(MODEL_DIR)
    metrics = {"base_model": BASE_MODEL, "epochs": EPOCHS, "dropout": DROPOUT, "seed": SEED,
               "train_ids": [r["id"] for r in train_records],
               "test_ids": [r["id"] for r in test_records],
               "baseline": baseline, "fine_tuned": results}
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nsaved model and metrics.json to {os.path.relpath(MODEL_DIR, ROOT)}/")
