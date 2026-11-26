"""Tests for the orchestrator's Agent 4 step (start_agent4 in agents/orchestrator.py, --interview).
Agent 4 itself is replaced by a fake here (its own tests are tests/test_interview_agent.py)."""
import interview_agent
import orchestrator
from interview_questions import BriefError

CHOSEN = {"selected": {"title": "Data Analyst"}, "handoff": {"brief": "/tmp/output/interview_brief.json"}}
QUIT = {"selected": None, "handoff": None}
RESULTS = {"overall": {"questions_answered": 4, "questions_total": 5, "speech": None, "vision": None}, "questions": []}


def test_not_requested_never_starts_agent4(monkeypatch):
    monkeypatch.setattr(interview_agent, "run_interview", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no")))
    assert orchestrator.start_agent4(CHOSEN, requested=False) is None


def test_quitting_at_the_job_choice_never_starts_agent4(monkeypatch, capsys):
    monkeypatch.setattr(interview_agent, "run_interview", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no")))
    assert orchestrator.start_agent4(QUIT, requested=True) is None
    assert "interview was not started" in capsys.readouterr().out


def test_a_chosen_job_starts_agent4_with_this_runs_brief(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(interview_agent, "run_interview", lambda brief, **kwargs: calls.append(brief) or RESULTS)
    assert orchestrator.start_agent4(CHOSEN, requested=True) is RESULTS
    assert calls == ["/tmp/output/interview_brief.json"]
    assert "Answered 4 of 5 questions" in capsys.readouterr().out


def test_a_brief_problem_is_a_warning_not_a_crash(monkeypatch, capsys):
    def refuse(brief, **kwargs):
        raise BriefError("The interview brief is older than the latest orchestrator run")
    monkeypatch.setattr(interview_agent, "run_interview", refuse)
    assert orchestrator.start_agent4(CHOSEN, requested=True) is None
    assert "Agent 4 could not start" in capsys.readouterr().err


def test_keep_recordings_is_passed_to_agent4(monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(interview_agent, "run_interview",
                        lambda brief, **kwargs: seen.append(kwargs) or RESULTS)
    orchestrator.start_agent4(CHOSEN, requested=True)
    orchestrator.start_agent4(CHOSEN, requested=True, keep_recordings=True)
    assert seen == [{"keep_recordings": False}, {"keep_recordings": True}]
