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


# --- C3: raw/ is a route to a knowable number too, not just results/ ---
#
# `--verify` proves every published measurement is a deterministic function
# of raw/trials.jsonl and claim.yaml. So once raw/ is committed the numbers
# are knowable even before results/ exists, and the threshold could still be
# moved afterwards: commit claim.yaml -> commit raw/trials.jsonl -> edit the
# threshold -> commit results/. A check that only watches results/ misses
# this entirely.


def test_claim_edited_after_raw_alone_is_reported_even_with_no_results_yet(
    repo: Path,
):
    _commit(repo, "findings/f/claim.yaml", "claim: x", "pre-register f")
    _commit(repo, "findings/f/raw/trials.jsonl", "{}", "raw trials for f")
    _commit(repo, "findings/f/claim.yaml", "claim: y  # loosened", "edit claim after seeing raw trials")
    problems = check(repo, "f")
    assert problems
    assert "claim.yaml was modified after results existed" in problems[0]


def test_claim_untouched_after_raw_alone_is_fine(repo: Path):
    _commit(repo, "findings/f/claim.yaml", "claim: x", "pre-register f")
    _commit(repo, "findings/f/raw/trials.jsonl", "{}", "raw trials for f")
    assert check(repo, "f") == []


def test_raw_committed_before_the_claim_is_reported(repo: Path):
    _commit(repo, "findings/f/raw/trials.jsonl", "{}", "raw trials first")
    _commit(repo, "findings/f/claim.yaml", "claim: x", "claim afterwards")
    problems = check(repo, "f")
    assert problems and "before" in problems[0]


def test_the_exact_attack_sequence_raw_then_edit_then_results_is_caught(
    repo: Path,
):
    """commit claim -> commit raw/ -> edit the threshold -> commit results/.

    Watching only results/ (the pre-fix behaviour) reports this clean: the
    claim was committed before results/. It was not committed before raw/,
    and raw/trials.jsonl is all --verify needs to know every number, so the
    threshold was demonstrably moved after the numbers were knowable.
    """
    _commit(repo, "findings/f/claim.yaml",
            "claim: x\nthreshold: 0.10", "pre-register f")
    _commit(repo, "findings/f/raw/trials.jsonl", "{}", "raw trials for f")
    _commit(repo, "findings/f/claim.yaml",
            "claim: x\nthreshold: 0.02  # loosened after seeing raw trials",
            "quietly loosen the threshold")
    _commit(repo, "findings/f/results/measurements.json", "{}",
            "results for f")
    problems = check(repo, "f")
    assert problems
    assert "claim.yaml was modified after results existed" in problems[0]


def test_raw_alone_does_not_falsely_flag_a_claim_that_precedes_it(repo: Path):
    """A model with only raw/ committed so far (results/ not written yet)
    must not be penalised just because raw/ exists -- only an actual
    post-raw edit to claim.yaml is a violation."""
    _commit(repo, "findings/f/claim.yaml", "claim: x", "pre-register f")
    _commit(repo, "findings/f/raw/trials.jsonl", "{}", "raw trials for f")
    _commit(repo, "findings/f/results/measurements.json", "{}",
            "results for f, still consistent with the original claim")
    assert check(repo, "f") == []


# --- C4: fail CLOSED, never silently report success for something unchecked ---


def test_an_unknown_finding_name_is_reported_not_silently_ok(repo: Path):
    # findings/f exists (from the fixture); findings/does-not-exist does not.
    _commit(repo, "findings/f/claim.yaml", "claim: x", "pre-register f")
    problems = check(repo, "does-not-exist")
    assert problems
    assert "does-not-exist" in problems[0]


def test_a_directory_with_no_git_history_is_reported_not_silently_ok(
    tmp_path: Path,
):
    # A tarball export: claim.yaml and results/ both present on disk, but no
    # .git at all -- nothing for this check to verify anything against.
    p = tmp_path / "findings" / "f"
    p.mkdir(parents=True)
    (p / "claim.yaml").write_text("claim: x", encoding="utf-8")
    (p / "results").mkdir()
    (p / "results" / "measurements.json").write_text("{}", encoding="utf-8")
    problems = check(tmp_path, "f")
    assert problems
    assert "git history" in problems[0]


def test_a_git_repo_with_zero_commits_is_reported_not_silently_ok(
    tmp_path: Path,
):
    _run(tmp_path, "init", "-q")
    _run(tmp_path, "config", "user.email", "t@example.com")
    _run(tmp_path, "config", "user.name", "T")
    p = tmp_path / "findings" / "f"
    p.mkdir(parents=True)
    (p / "claim.yaml").write_text("claim: x", encoding="utf-8")
    problems = check(tmp_path, "f")
    assert problems
    assert "git history" in problems[0]
