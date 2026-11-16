"""The 3 fake resumes in unseen layouts (tests/fixtures/unseen/) and the
checker that lists what Agent 1 gets wrong on them
(scripts/check_unseen_layouts.py). The known breakages are listed in the
README, not asserted here; these tests only keep the files and the checker working."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "tests", "fixtures"))

from check_unseen_layouts import compare  # noqa: E402
from make_unseen_layouts import EXPECTED, OUT  # noqa: E402
from resume_parser import read_resume  # noqa: E402


def test_unseen_resumes_exist_and_are_readable():
    for filename, expected in EXPECTED.items():
        text = read_resume(os.path.join(OUT, filename))
        assert expected["email"] in text


def test_two_column_pdf_really_has_two_columns():
    # The reader merges the columns line by line: that IS the unseen layout
    text = read_resume(os.path.join(OUT, "two_column.pdf"))
    assert "Python, SQL, Docker Harborline Logistics | IT Support Specialist" in text


def test_compare_marks_a_correct_profile_all_right():
    expected = EXPECTED["season_dates.docx"]
    profile = {"name": expected["name"], "email": expected["email"],
               "skills": list(reversed(expected["skills"])),            # order doesn't matter
               "education": dict(expected["education"]),
               "experience": [{"company": c, "title": t, "months": m} for c, t, m in expected["experience"]],
               "projects": []}
    assert all(ok for _, ok, _, _ in compare(profile, expected))
    profile["experience"] = profile["experience"][:1]
    assert [f for f, ok, _, _ in compare(profile, expected) if not ok] == ["experience"]
