"""A prediction written after seeing the numbers is not a prediction.

This is the only claim this repository makes that a reader cannot check by
re-running the analysis, so it is checked mechanically against git history
instead of being promised in prose.
"""

import subprocess
from pathlib import Path

import pytest

from tools.check_preregistration import check, first_commit_date

REPO = Path(__file__).resolve().parents[1]


def _run(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True,
                   capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _run(tmp_path, "init", "-q")
    _run(tmp_path, "config", "user.email", "t@example.com")
    _run(tmp_path, "config", "user.name", "T")
    (tmp_path / "findings" / "f").mkdir(parents=True)
    return tmp_path


def _commit(repo: Path, rel: str, body: str, message: str) -> None:
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    _run(repo, "add", rel)
    _run(repo, "commit", "-q", "-m", message)


def _delete(repo: Path, rel: str, message: str) -> None:
    (repo / rel).unlink()
    _run(repo, "add", rel)
    _run(repo, "commit", "-q", "-m", message)


def test_first_commit_date_is_none_for_an_uncommitted_file(repo: Path):
    (repo / "findings" / "f" / "claim.yaml").write_text("x", encoding="utf-8")
    assert first_commit_date(repo, "findings/f/claim.yaml") is None


def test_a_claim_committed_before_its_results_passes(repo: Path):
    _commit(repo, "findings/f/claim.yaml", "claim: x", "pre-register f")
    _commit(repo, "findings/f/results/measurements.json", "{}", "results for f")
    assert check(repo, "f") == []


def test_results_committed_in_the_same_commit_as_the_claim_is_reported(repo: Path):
    p = repo / "findings" / "f"
    (p / "claim.yaml").write_text("claim: x", encoding="utf-8")
    (p / "results").mkdir()
    (p / "results" / "measurements.json").write_text("{}", encoding="utf-8")
    _run(repo, "add", ".")
    _run(repo, "commit", "-q", "-m", "both at once")
    problems = check(repo, "f")
    assert problems and "same commit" in problems[0]


def test_results_committed_before_the_claim_is_reported(repo: Path):
    _commit(repo, "findings/f/results/measurements.json", "{}", "results first")
    _commit(repo, "findings/f/claim.yaml", "claim: x", "claim afterwards")
    problems = check(repo, "f")
    assert problems and "before" in problems[0]


def test_a_missing_claim_is_reported(repo: Path):
    _commit(repo, "findings/f/results/measurements.json", "{}", "results only")
    problems = check(repo, "f")
    assert problems and "claim.yaml" in problems[0]


def test_results_not_yet_committed_is_fine(repo: Path):
    _commit(repo, "findings/f/claim.yaml", "claim: x", "pre-register f")
    assert check(repo, "f") == []


# --- The claim must be frozen once results exist, not merely predate them ---
#
# first_commit_date only sees the commit that first ADDED a path. A claim that
# is edited after its results exist would slip past a check built only on
# that function: the edit is invisible to it. These tests cover the edit, and
# the equivalent delete-then-recreate, that a naive "was it added before
# results" check would miss.


def test_claim_edited_after_results_is_reported(repo: Path):
    _commit(repo, "findings/f/claim.yaml", "claim: x", "pre-register f")
    _commit(repo, "findings/f/results/measurements.json", "{}", "results for f")
    _commit(repo, "findings/f/claim.yaml", "claim: y  # loosened", "edit claim after seeing results")
    problems = check(repo, "f")
    assert problems
    assert "claim.yaml was modified after results existed" in problems[0]


def test_claim_untouched_after_results_is_fine(repo: Path):
    _commit(repo, "findings/f/claim.yaml", "claim: x", "pre-register f")
    _commit(repo, "findings/f/results/measurements.json", "{}", "results for f")
    assert check(repo, "f") == []


def test_claim_edited_before_results_is_fine(repo: Path):
    _commit(repo, "findings/f/claim.yaml", "claim: x", "pre-register f")
    _commit(repo, "findings/f/claim.yaml", "claim: x refined", "refine claim before results")
    _commit(repo, "findings/f/results/measurements.json", "{}", "results for f")
    assert check(repo, "f") == []


def test_claim_deleted_and_readded_after_results_is_reported(repo: Path):
    _commit(repo, "findings/f/claim.yaml", "claim: x", "pre-register f")
    _delete(repo, "findings/f/claim.yaml", "withdraw claim")
    _commit(repo, "findings/f/results/measurements.json", "{}", "results for f")
    _commit(repo, "findings/f/claim.yaml", "claim: z  # different", "re-add claim with different content")
    problems = check(repo, "f")
    assert problems
    assert "claim.yaml was modified after results existed" in problems[0]


def test_results_with_no_claim_at_all_is_still_reported(repo: Path):
    _commit(repo, "findings/f/results/measurements.json", "{}", "results only, no claim ever")
    problems = check(repo, "f")
    assert problems and "claim.yaml" in problems[0]
