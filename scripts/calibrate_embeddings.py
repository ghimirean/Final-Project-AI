"""Shows where EMBED_LOW / EMBED_HIGH in agents/job_synthesizer.py come from.

Embeds each resume against the 6 fake job postings AND 3 postings that have
nothing to do with software, then prints the similarity table. The
unrelated postings show what "no real match" looks like for this model;
the best real matches show what "strong match" looks like.

Run from project root (loads the real model; downloads it the first time):
    python scripts/calibrate_embeddings.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agents"))

from embeddings import EMBED_CONFIG, resume_job_similarities  # noqa: E402
from job_synthesizer import EMBED_HIGH, EMBED_LOW, SKILLS_PATH, similarity_to_points  # noqa: E402
from resume_parser import parse_resume  # noqa: E402

UNRELATED = {
    "chef": "Pastry chef wanted for a busy bakery. Prepare croissants, cakes and desserts "
            "daily. Culinary school diploma preferred. Early morning shifts.",
    "nurse": "Registered nurse for the pediatric ward. Administer medication, monitor "
             "patients, and coordinate with physicians. Nursing license required.",
    "truck": "Long-haul truck driver. Valid commercial license, clean driving record, "
             "able to lift 50 pounds.",
}
RESUMES = [
    "tests/fixtures/resume_title_case.docx",
    "tests/fixtures/resume_numeric_dates.docx",
    "tests/fixtures/resume_comma_skills.docx",
    "data/sample.pdf",
]

if __name__ == "__main__":
    with open(os.path.join(ROOT, "tests", "fixtures", "jobs.json"), encoding="utf-8") as f:
        jobs = json.load(f)
    labels = [j["id"][-3:] for j in jobs] + list(UNRELATED)
    descriptions = [j["description"] for j in jobs] + list(UNRELATED.values())

    print(f"Model: {EMBED_CONFIG['model']}   anchors: {EMBED_LOW} -> 0 pts, {EMBED_HIGH} -> 40 pts\n")
    print(f"{'':27}" + "".join(f"{label:>7}" for label in labels))
    for path in RESUMES:
        full_path = os.path.join(ROOT, path)
        if not os.path.exists(full_path):
            continue
        text = parse_resume(full_path, skills_path=SKILLS_PATH)["raw_text"]   # rules only, no API call
        similarities = resume_job_similarities(text, descriptions)
        name = os.path.basename(path)[:20]
        print(f"{name:<20} sim   " + "".join(f"{s:7.2f}" for s in similarities))
        print(f"{'':<20} pts   " + "".join(f"{similarity_to_points(s):7.1f}" for s in similarities))
