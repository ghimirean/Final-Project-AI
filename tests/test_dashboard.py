"""Tests for the HITL dashboard (dashboard.py).

Plain tests cover the non-UI logic. The UI tests drive the real page with
Streamlit's AppTest (streamlit.testing.v1) in OFFLINE mode only: no JSearch
or Groq call is ever made, and conftest.py blocks the models (AppTest runs
the script in this process, so those safety nets apply to it too). Output
goes to a temp folder via INTERVAI_OUTPUT_DIR. All data is fake.
"""
import json
import os
import sys
import tempfile

import pytest
from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import dashboard  # noqa: E402
import orchestrator  # noqa: E402
from dashboard import (CUSTOM, bullets, breakdown_rows, clear_run_state,  # noqa: E402
                       failure_message, file_extension, make_choice, md, mode_options,
                       role_problem, run_agents, safe_url, stage_notices)
from orchestrator import DEMO_REPLAY, MAX_STEPS, Trace  # noqa: E402

APP = os.path.join(ROOT, "dashboard.py")
RESUME = os.path.join(ROOT, "tests", "fixtures", "resume_title_case.docx")   # FAKE resume
CONFIRM = "Confirm and prepare the interview"
QUIT = "Quit without choosing"


# ---------- modes and the uploaded file ----------

def test_mode_options():
    assert mode_options("demo") == {"demo": True}
    assert mode_options("offline") == {"offline": True}
    assert mode_options("live") == {"max_steps": 1}          # live defaults to 1 step
    assert mode_options("live", 3) == {"max_steps": 3}


@pytest.mark.parametrize("steps", [0, MAX_STEPS + 1])
def test_live_mode_rejects_bad_step_counts(steps):
    with pytest.raises(ValueError, match="max_steps"):
        mode_options("live", steps)


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown mode"):
        mode_options("turbo")


def test_file_extension():
    assert file_extension("CV.PDF") == ".pdf"
    assert file_extension("resume.docx") == ".docx"
    for bad in ("resume.txt", "resume", None):
        with pytest.raises(ValueError, match=".pdf or .docx"):
            file_extension(bad)


def test_run_agents_deletes_the_resume_after_use(tmp_path, monkeypatch):
    monkeypatch.setenv("INTERVAI_OUTPUT_DIR", str(tmp_path / "out"))
    seen = {}

    def fake_prepare(path, location, **kwargs):
        with open(path, "rb") as f:
            seen.update(path=path, data=f.read(), location=location, kwargs=kwargs)
        return {"ranked_jobs": []}

    run_agents("cv.docx", b"fake bytes", "  Remote ", "offline", prepare=fake_prepare)
    assert seen["data"] == b"fake bytes" and seen["path"].endswith(".docx")
    assert not os.path.exists(seen["path"])                  # deleted right after
    assert seen["location"] == "Remote"
    assert seen["kwargs"]["offline"] is True
    assert seen["kwargs"]["out_dir"] == str(tmp_path / "out")


def test_run_agents_deletes_the_resume_even_if_an_agent_fails():
    seen = {}

    def broken(path, location, **kwargs):
        seen["path"] = path
        raise RuntimeError("Agent 1 broke")

    with pytest.raises(RuntimeError, match="Agent 1 broke"):
        run_agents("cv.pdf", b"x", "", "offline", prepare=broken)
    assert not os.path.exists(seen["path"])


def test_run_agents_writes_nothing_for_a_bad_file_type():
    def never(*args, **kwargs):
        raise AssertionError("prepare must not run")
    with pytest.raises(ValueError):
        run_agents("cv.exe", b"x", "", "offline", prepare=never)


def test_clear_run_state_drops_the_run_and_the_upload():
    session = {"run": {"profile": {"raw_text": "resume text"}}, "picked": 2, "role": "QA",
               "result": {"title": "x"}, "upload_key": 4}
    clear_run_state(session)
    assert session == {"result": {"title": "x"}, "upload_key": 5}
    empty = {}
    clear_run_state(empty)
    assert empty == {"upload_key": 1}


# ---------- untrusted text and links ----------

def test_md_escapes_markdown_and_html():
    out = md("[click](javascript:alert(1)) <b>*bold*</b> `code` $x$")
    for raw in ("[", "]", "(", ")", "<", ">", "*", "`", "$"):
        assert raw not in out.replace("\\" + raw, "")       # every one is escaped


def test_md_makes_one_short_line():
    assert md("a\n\nb\tc") == "a b c"
    assert md("x" * 500, limit=10) == "x" * 9 + "…"
    assert md(None) == ""


@pytest.mark.parametrize("url", ["https://example.com/job?id=1", "http://x.test/a",
                                 "HTTPS://Example.com/J"])
def test_safe_url_accepts_http_and_https(url):
    assert safe_url(url) == url


@pytest.mark.parametrize("url", ["javascript:alert(1)", "data:text/html,<b>x</b>",
                                 "ftp://x.test/a", "/jobs/1", "www.example.com",
                                 "https://x.test/a b", "https://x.test/\nnext", "", None, 42])
def test_safe_url_rejects_everything_else(url):
    assert safe_url(url) is None


def test_bullets():
    assert bullets(["Python", "*SQL*"]) == "- Python\n- \\*SQL\\*"
    assert bullets([]) == "_none_"


def test_breakdown_rows():
    rows = breakdown_rows({"skill_points": 30.0, "text_points": 20.0, "text_method": "embeddings",
                           "text_similarity": 0.4, "experience_penalty": 10, "required_years": 3,
                           "candidate_years": 2, "degree_penalty": 5,
                           "required_degree": "Master", "candidate_degree": "Bachelor"})
    assert [(p, v) for p, v, _ in rows] == [("Skill overlap", 30.0), ("Text similarity", 20.0),
                                            ("Experience gap", -10), ("Degree gap", -5)]
    assert "needs Master" in rows[3][2]


# ---------- the choice ----------

RANKED = [{"id": "a", "title": "Data Analyst", "score": 80.0},
          {"id": "b", "title": "QA Tester", "score": 40.0}]


def test_make_choice_returns_the_ranked_job():
    assert make_choice(RANKED, 1) is RANKED[1]


def test_make_choice_custom_role():
    assert make_choice(RANKED, CUSTOM, "  Data Engineer ") == {"custom_role": "Data Engineer"}


@pytest.mark.parametrize("role", ["", "   ", "1234", "x" * 81])
def test_bad_custom_roles_are_rejected(role):
    assert role_problem(role)
    with pytest.raises(ValueError):
        make_choice(RANKED, CUSTOM, role)


def test_make_choice_rejects_a_missing_job():
    with pytest.raises(ValueError):
        make_choice(RANKED, 5)


# ---------- loading and error messages ----------

def fake_state(source, reason=None, ranked=True, **details):
    trace = Trace()
    trace.stage("agent1", 0)
    trace.stage("agent2", 0, fallback=reason, jobs_source=source, **details)
    trace.stage("agent3", 0)
    return {"trace": trace, "jobs_source": source, "ranked_jobs": RANKED if ranked else []}


def levels(notices):
    return [level for level, _ in notices]


def test_notice_for_offline_mode():
    notices = stage_notices(fake_state("fixture", "offline mode: fixture jobs used on purpose"),
                            "offline")
    assert levels(notices) == ["info"] and "Offline mode" in notices[0][1]


def test_notice_for_demo_without_saved_jobs():
    notices = stage_notices(fake_state("fixture", "demo replay found no saved real jobs"), "demo")
    assert levels(notices) == ["warning"] and "no saved real jobs" in notices[0][1]


def test_notice_for_demo_replay():
    notices = stage_notices(fake_state(DEMO_REPLAY, "demo replay: ...", real_jobs=9,
                                       saved="2026-11-16T18:00:00"), "demo")
    assert levels(notices) == ["info"] and "replaying 9 real jobs" in notices[0][1]


@pytest.mark.parametrize("reason", [
    "Agent 2's searches failed (HTTP 429: JSearch rate limit or monthly quota reached)",
    "Agent 2 failed (HTTP 429): quota"])
def test_notice_when_the_job_source_is_out_of_requests(reason):
    notices = stage_notices(fake_state("fixture", reason), "live")
    assert levels(notices) == ["error"] and "out of requests" in notices[0][1]


def test_notice_for_a_missing_key():
    notices = stage_notices(fake_state("fixture", "no JSearch key (RAPIDAPI_KEY not set in .env)"),
                            "live")
    assert "RAPIDAPI_KEY" in notices[0][1]


def test_notice_when_the_live_search_finds_nothing():
    notices = stage_notices(fake_state("fixture", "Agent 2 found no jobs (1 searches, ...)"), "live")
    assert levels(notices) == ["warning"] and "found no jobs" in notices[0][1]


def test_notice_for_another_search_failure():
    notices = stage_notices(fake_state("fixture", "Agent 2 failed: network error"), "live")
    assert levels(notices) == ["error"] and "network error" in notices[0][1]


def test_no_notice_for_a_clean_live_search():
    assert stage_notices(fake_state("jsearch"), "live") == []


def test_notice_when_there_are_no_jobs_at_all():
    notices = stage_notices(fake_state("jsearch", ranked=False), "live")
    assert levels(notices) == ["warning"] and "No jobs" in notices[0][1]


def test_failure_message_names_the_stage():
    assert failure_message("agent1", ValueError("bad PDF")).startswith("Agent 1: resume parser failed")
    assert "Could not start" in failure_message(None, ValueError("bad"))


# ---------- the page (AppTest, offline mode) ----------

@pytest.fixture
def app_env(tmp_path, monkeypatch):
    """Temp output folder and temp-file folder; counts prepare_run calls."""
    out, tmp = tmp_path / "out", tmp_path / "tmp"
    tmp.mkdir()
    monkeypatch.setenv("INTERVAI_OUTPUT_DIR", str(out))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp))
    calls = []
    real = orchestrator.prepare_run

    def counting(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)
    monkeypatch.setattr(orchestrator, "prepare_run", counting)
    return {"out": out, "tmp": tmp, "calls": calls}


def new_app():
    return AppTest.from_file(APP, default_timeout=120).run()


def find_jobs(at, name="cv.docx", data=None, mode="offline"):
    if data is None:
        with open(RESUME, "rb") as f:
            data = f.read()
    at.file_uploader[0].set_value((name, data, "application/octet-stream")).run()
    at.sidebar.radio[0].set_value(mode).run()
    at.sidebar.button[0].click().run()
    return at


def button(at, label):
    return next(b for b in at.button if b.label == label)


def resume_temp_files(env):
    return [f for f in os.listdir(env["tmp"]) if f.startswith("intervai_resume_")]


def test_start_page(app_env):
    at = new_app()
    assert not at.exception
    assert at.sidebar.radio[0].value == "demo"
    assert at.sidebar.button[0].disabled                 # no resume yet
    assert app_env["calls"] == []


def test_live_mode_defaults_to_one_step_with_a_warning(app_env):
    at = new_app()
    at.sidebar.radio[0].set_value("live").run()
    assert at.sidebar.number_input[0].value == 1
    assert any("spends JSearch requests" in w.value for w in at.sidebar.warning)
    assert app_env["calls"] == []                        # nothing runs until "Find jobs"


def test_find_jobs_shows_cards_and_runs_the_agents_once(app_env):
    at = find_jobs(new_app())
    assert not at.exception
    assert len(app_env["calls"]) == 1
    titles = [m.value for m in at.markdown if m.value.startswith("#### ")]
    assert len(titles) == 6
    assert len(at.expander) == 6                         # a score breakdown per card
    assert any("Offline mode" in i.value for i in at.info)
    assert resume_temp_files(app_env) == []              # temp resume already deleted
    at.run()                                             # a plain rerun ...
    at.radio(key="picked").set_value(1).run()            # ... or picking a job ...
    assert len(app_env["calls"]) == 1                    # ... never runs the agents again


def test_confirm_writes_the_brief_and_clears_the_run(app_env):
    at = find_jobs(new_app())
    at.radio(key="picked").set_value(1).run()
    chosen = at.session_state["run"]["ranked_jobs"][1]
    button(at, CONFIRM).click().run()
    assert not at.exception
    with open(app_env["out"] / "interview_brief.json", encoding="utf-8") as f:
        brief = json.load(f)
    assert brief["job"]["id"] == chosen["id"]
    with open(app_env["out"] / "run_trace.json", encoding="utf-8") as f:
        agents = [s["agent"] for s in json.load(f)["stages"]]
    assert agents[-2:] == ["Human in the loop: job choice", "Agent 4: interview (handoff)"]
    assert "run" not in at.session_state                 # resume text is gone
    assert at.file_uploader[0].value is None             # and so is the upload
    assert at.success
    assert resume_temp_files(app_env) == []


def test_custom_role(app_env):
    at = find_jobs(new_app())
    at.radio(key="picked").set_value(CUSTOM).run()
    assert any("role name only" in c.value for c in at.caption)
    button(at, CONFIRM).click().run()
    assert any("Type the role" in e.value for e in at.error)
    assert not (app_env["out"] / "interview_brief.json").exists()
    at.text_input(key="role").input("Data Engineer").run()
    button(at, CONFIRM).click().run()
    with open(app_env["out"] / "interview_brief.json", encoding="utf-8") as f:
        assert json.load(f)["job"]["title"] == "Data Engineer"


def test_quit_writes_no_brief(app_env):
    at = find_jobs(new_app())
    button(at, QUIT).click().run()
    assert not (app_env["out"] / "interview_brief.json").exists()
    with open(app_env["out"] / "run_trace.json", encoding="utf-8") as f:
        assert json.load(f)["stages"][-1]["details"]["quit"] is True
    assert any("no interview brief" in i.value for i in at.info)
    assert "run" not in at.session_state
    button(at, "Start over").click().run()
    assert any("Upload a resume" in i.value for i in at.info)


def test_a_broken_resume_shows_an_error_and_is_deleted(app_env):
    at = find_jobs(new_app(), name="broken.pdf", data=b"not a real pdf")
    assert not at.exception
    assert any(e.value.startswith("Agent 1: resume parser failed") for e in at.error)
    assert "run" not in at.session_state
    assert resume_temp_files(app_env) == []


def test_untrusted_job_text_and_links_are_not_rendered(app_env, tmp_path, monkeypatch):
    jobs = [{"id": "evil", "title": "[Click me](javascript:alert(1)) <script>x</script>",
             "company": "**Bad** Co", "location": "Remote", "url": "javascript:alert(1)",
             "description": "Python and SQL work."},
            {"id": "good", "title": "Data Analyst", "company": "Acme", "location": "Remote",
             "url": "https://example.com/jobs/1", "description": "Python and SQL work."}]
    path = tmp_path / "jobs.json"
    path.write_text(json.dumps(jobs), encoding="utf-8")
    monkeypatch.setattr(orchestrator, "FIXTURE_JOBS", str(path))
    at = find_jobs(new_app())
    assert not at.exception
    text = "\n".join(m.value for m in at.markdown)
    assert "\\[Click me\\]\\(javascript\\:alert\\(1\\)\\)" in text   # shown as plain text
    assert "<script>" not in text and "\\<script\\>" in text
    links = [el.proto.url for el in at.get("link_button")]
    assert links == ["https://example.com/jobs/1"]                  # only the safe one
    assert any("No safe link" in c.value for c in at.caption)
