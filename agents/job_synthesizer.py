"""Agent 3: job description synthesizer + match scoring.

JOB INPUT FORMAT
----------------
Agent 3 takes a list of job postings. Each posting is a dict:

    {
        "id":          "job-001",                 # unique id (string)
        "title":       "Junior Java Developer",
        "company":     "Maplewood Software",
        "location":    "Monroe, LA",
        "url":         "https://example.com/jobs/job-001",
        "description": "Full job description text ..."
    }

All values are strings. Only "description" is used to find requirements;
the other fields are passed through so they can be shown to the user.
An example list lives in tests/fixtures/jobs.json.
"""
import json
import os
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from resume_parser import load_skills, extract_skills

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILLS_PATH = os.path.join(ROOT, "data", "skills.txt")


def load_jobs(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# Matches "5+ years", "2 years", "at least 1 year", "3 or more years"
YEARS_PATTERN = r"(\d+)\s*\+?\s*(?:or more\s+)?years?\b"

# Degree levels from lowest to highest, with the words that signal each one
DEGREE_LEVELS = [
    ("Associate", r"\bAssociate'?s?\b"),
    ("Bachelor", r"\bBachelor'?s?\b|\bB\.?S\.?\b|\bB\.?A\.?\b"),
    ("Master", r"\bMaster'?s?\b|\bM\.?S\.?\b|\bMBA\b"),
    ("PhD", r"\bPh\.?D\.?\b|\bDoctorate\b"),
]


def extract_min_years(text):
    # Takes the largest number found, e.g. "5+ years" -> 5. None found -> 0.
    numbers = [int(n) for n in re.findall(YEARS_PATTERN, text, re.IGNORECASE)]
    return max(numbers) if numbers else 0


def extract_degree(text):
    # Returns the LOWEST degree mentioned, since "Bachelor's or Master's"
    # means a Bachelor's is enough. None if no degree is mentioned.
    for level, pattern in DEGREE_LEVELS:
        if re.search(pattern, text):
            return level
    return None


def extract_requirements(job, skill_list=None):
    """Reads a job's description and returns what it asks for:
    {"skills": [...], "min_years": int, "degree": str or None}
    """
    if skill_list is None:
        skill_list = load_skills(SKILLS_PATH)
    text = job["description"]
    return {
        "skills": extract_skills(text, skill_list),
        "min_years": extract_min_years(text),
        "degree": extract_degree(text),
    }


# ---------- Match scoring ----------

# WEIGHTS (total = 100 points)
# - Skills (60 pts): the most direct signal. Recruiters screen by "do you
#   have the listed skills", so the share of required skills you have
#   counts the most.
# - Text similarity (40 pts): catches overlap the skill list misses
#   (job titles, field of study, words like "backend" or "data"). It is
#   noisier than an exact skill match, so it gets less weight. Raw TF-IDF
#   similarity is small (about 0.1-0.3), so it is rescaled: the job most
#   similar to the resume gets the full 40, the others get a share of it.
# - Experience penalty: minus 10 pts per year you are short of the job's
#   minimum, capped at 30, so a senior role drops down the list without
#   being thrown out completely.
# - Degree penalty: minus 5 if the job asks for a higher degree than you
#   have. Small, because students often apply while still finishing one.
SKILL_WEIGHT = 60
TEXT_WEIGHT = 40
PENALTY_PER_YEAR = 10
MAX_PENALTY = 30
DEGREE_PENALTY = 5

DEGREE_ORDER = [level for level, _ in DEGREE_LEVELS]   # Associate < ... < PhD


def degree_rank(level):
    # Position in DEGREE_ORDER; no degree counts as -1 (below Associate)
    return DEGREE_ORDER.index(level) if level in DEGREE_ORDER else -1


def resume_text(profile):
    # Prefer the full resume text from Agent 1; fall back to a short summary
    return profile.get("raw_text") or profile_to_text(profile)


def profile_to_text(profile):
    # Short summary for profiles that don't carry raw_text
    parts = list(profile.get("skills", []))
    for job in profile.get("experience", []):
        parts += [job.get("title") or "", job.get("company") or ""]
    education = profile.get("education") or {}
    parts += [education.get("degree") or "", education.get("field") or ""]
    return " ".join(p for p in parts if p)


def job_similarities(text, descriptions):
    """TF-IDF turns each text into a vector of word weights. It is fitted on
    the resume AND all job descriptions together, so words that appear in
    every posting ("team", "experience") count less than specific ones
    ("PyTorch", "backend"). Cosine similarity then measures how much the
    resume vector points the same way as each job vector (0 = nothing shared).

    Returns (raw, scaled): raw cosine values, and values divided by the
    largest one so the best-matching job gets 1.0.
    """
    vectors = TfidfVectorizer(stop_words="english").fit_transform([text] + descriptions)
    raw = cosine_similarity(vectors[0], vectors[1:])[0].tolist()
    best = max(raw) if raw else 0.0
    scaled = [r / best if best > 0 else 0.0 for r in raw]
    return raw, scaled


def score_match(profile, job, skill_list=None, similarity=None):
    """Scores how well a resume profile fits one job, from 0 to 100.
    Returns {"score": float, "breakdown": {...}} so the score can be explained.

    similarity: optional (raw, scaled) pair from job_similarities(). rank_jobs
    passes it in so jobs are compared against each other. When scoring a
    single job on its own, it is computed here, and that job counts as the
    best match (scaled = 1.0) if it shares any words with the resume.
    """
    req = extract_requirements(job, skill_list)
    have = set(profile.get("skills", []))
    matched = [s for s in req["skills"] if s in have]
    missing = [s for s in req["skills"] if s not in have]

    skill_ratio = len(matched) / len(req["skills"]) if req["skills"] else 0.0
    if similarity is None:
        raw, scaled = job_similarities(resume_text(profile), [job["description"]])
        similarity = (raw[0], scaled[0])
    raw_similarity, scaled_similarity = similarity

    years = profile.get("experience_years", 0)
    years_short = max(0, req["min_years"] - years)
    penalty = min(MAX_PENALTY, PENALTY_PER_YEAR * years_short)

    candidate_degree = extract_degree((profile.get("education") or {}).get("degree") or "")
    degree_short = req["degree"] is not None and degree_rank(req["degree"]) > degree_rank(candidate_degree)
    degree_penalty = DEGREE_PENALTY if degree_short else 0

    skill_points = SKILL_WEIGHT * skill_ratio
    text_points = TEXT_WEIGHT * scaled_similarity
    score = max(0.0, min(100.0, skill_points + text_points - penalty - degree_penalty))

    return {
        "score": round(score, 1),
        "breakdown": {
            "matched_skills": matched,
            "missing_skills": missing,
            "skill_points": round(skill_points, 1),
            "text_similarity": round(raw_similarity, 3),
            "text_points": round(text_points, 1),
            "required_years": req["min_years"],
            "candidate_years": years,
            "experience_penalty": round(penalty, 1),
            "required_degree": req["degree"],
            "candidate_degree": candidate_degree,
            "degree_penalty": degree_penalty,
        },
    }


# ---------- Ranking + output ----------

def make_reason(breakdown):
    # One readable line explaining the score, e.g.
    # "Matches 5/7 skills; missing GitHub, Linux; needs 5+ yrs (you have 2.6)."
    matched = breakdown["matched_skills"]
    missing = breakdown["missing_skills"]
    total = len(matched) + len(missing)
    if total == 0:
        parts = ["No known skills found in the posting"]
    else:
        parts = [f"Matches {len(matched)}/{total} skills"]
    if missing:
        shown = ", ".join(missing[:3])
        parts.append("missing " + shown + (" and more" if len(missing) > 3 else ""))
    if breakdown["experience_penalty"] > 0:
        parts.append(f"needs {breakdown['required_years']}+ yrs "
                     f"(you have {breakdown['candidate_years']})")
    if breakdown.get("degree_penalty", 0) > 0:
        parts.append(f"asks for a {breakdown['required_degree']}'s degree")
    return "; ".join(parts) + "."


def rank_jobs(profile, jobs):
    """Scores every job for this profile and returns them best-first.
    Each result keeps the job's fields and adds score, matched_skills,
    missing_skills, reason, and the full breakdown.
    """
    skill_list = load_skills(SKILLS_PATH)      # load once, not once per job
    raw, scaled = job_similarities(resume_text(profile), [j["description"] for j in jobs])
    ranked = []
    for job, r, s in zip(jobs, raw, scaled):
        result = score_match(profile, job, skill_list, similarity=(r, s))
        b = result["breakdown"]
        ranked.append({
            "id": job["id"],
            "title": job["title"],
            "company": job["company"],
            "location": job["location"],
            "url": job["url"],
            "score": result["score"],
            "matched_skills": b["matched_skills"],
            "missing_skills": b["missing_skills"],
            "reason": make_reason(b),
            "breakdown": b,
        })
    ranked.sort(key=lambda r: r["score"], reverse=True)
    return ranked


def print_table(ranked):
    print(f"{'#':<3}{'Score':>6}  {'Title':<30}{'Company':<28}Reason")
    print("-" * 110)
    for i, r in enumerate(ranked, 1):
        print(f"{i:<3}{r['score']:>6.1f}  {r['title'][:29]:<30}"
              f"{r['company'][:27]:<28}{r['reason']}")


def save_json(data, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import sys
    from resume_parser import parse_resume

    resume_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "sample.pdf")
    jobs_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "tests", "fixtures", "jobs.json")
    out_path = os.path.join(ROOT, "output", "ranked_jobs.json")

    profile = parse_resume(resume_path, skills_path=SKILLS_PATH)
    ranked = rank_jobs(profile, load_jobs(jobs_path))
    print_table(ranked)
    save_json(ranked, out_path)
    print(f"\nSaved {len(ranked)} ranked jobs to output/ranked_jobs.json")
