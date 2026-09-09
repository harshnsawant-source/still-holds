"""Prove the prediction was written before the numbers were seen.

WHY THIS IS MECHANICAL
----------------------
Every other claim in this repository is checkable by re-running the analysis
over committed raw outputs. This one is not: nobody can tell from a file
whether its prediction was written before or after the result. Git can, so git
is asked rather than the author trusted.

WHAT IT REFUSES TO DO
---------------------
Repair. It reports problems and returns them; it never rewrites history, never
amends a claim, and never decides that a violation is acceptable.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


class PreRegistrationError(Exception):
    """Raised by the CLI when a finding's pre-registration does not hold."""


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, capture_output=True,
                            text=True)
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def first_commit_date(repo: Path, relative_path: str) -> str | None:
    """ISO date of the commit that first added `relative_path`, or None.

    None means the path has never been committed, which is a normal state for
    results that have not been published yet.
    """
    out = _git(repo, "log", "--diff-filter=A", "--format=%H %cI", "--",
               relative_path)
    if not out:
        return None
    return out.splitlines()[-1].strip()


def last_commit_touching(repo: Path, relative_path: str) -> str | None:
    """"sha date" of the most recent commit that touched `relative_path`, or None.

    Unlike `first_commit_date`, this is not filtered to additions: an edit, or
    a delete followed by a re-add with different content, counts. It exists
    because a claim is not "pre-registered" merely because *some* commit of it
    predates the results — it is pre-registered only if *every* commit of it
    does. A claim rewritten after the numbers were seen, even quietly, must be
    caught here; `git log` on the bare path (no `--diff-filter`) surfaces that
    edit as the newest entry regardless of what happened to the path in
    between.
    """
    out = _git(repo, "log", "--format=%H %cI", "--", relative_path)
    if not out:
        return None
    return out.splitlines()[0].strip()


def _commit_order(repo: Path) -> list[str]:
    """All commit hashes reachable from HEAD, oldest first.

    Used to order two commits precisely. Committer-date strings are only
    accurate to the second, so two commits made in quick succession (exactly
    what an automated pre-registration commit followed by an automated
    results commit looks like) can carry the same timestamp and be
    unorderable by string comparison alone. Topological order from git's own
    parent pointers has no such gap.
    """
    out = _git(repo, "log", "--format=%H", "--reverse")
    return out.splitlines() if out else []


def _first_artifact_commit(repo: Path, finding: str) -> tuple[str, str] | None:
    """The earliest commit that first added EITHER `raw/` or `results/`.

    Both directories are a route to a knowable number: `--verify` proves
    every published measurement is a deterministic function of
    `raw/trials.jsonl`, so once `raw/` is committed the numbers are knowable
    even if `results/` has not been written yet — and the threshold could
    still be moved afterwards. Watching only `results/` let exactly that
    sequence through: commit claim.yaml, commit raw/trials.jsonl, edit the
    threshold, commit results/. The claim must be frozen as soon as EITHER
    path first appears, not whichever one this function's caller happened to
    remember to watch.

    Returns `(which, "sha date")` for whichever of "raw" / "results" was
    committed first, by commit-graph order rather than committer-date
    string (see `_commit_order`'s docstring for why); `None` if neither has
    ever been committed.
    """
    candidates = []
    for which in ("raw", "results"):
        commit = first_commit_date(repo, f"findings/{finding}/{which}")
        if commit is not None:
            candidates.append((which, commit))
    if not candidates:
        return None
    order = _commit_order(repo)

    def _rank(item: tuple[str, str]) -> int:
        sha = item[1].split(" ", 1)[0]
        return order.index(sha) if order and sha in order else -1

    candidates.sort(key=_rank)
    return candidates[0]


def check(repo: Path, finding: str) -> list[str]:
    """Return the pre-registration problems for one finding. Empty means fine.

    The rule enforced is not "the claim was added before the results" — it is
    that the claim's LAST modification precedes the FIRST appearance of
    EITHER `raw/` or `results/`, strictly (see `_first_artifact_commit`).
    Once either exists for a finding, its claim.yaml is frozen: any further
    commit that touches it, by any means (edit, or delete-then-recreate), is
    a violation, because that is precisely the
    quiet-loosening-of-a-threshold-after-seeing-the-number this check exists
    to catch. There is no "small correction" exemption. A claim that
    genuinely needs to change after results exist should become a new claim
    in a new commit, with the old results republished under it — a decision
    for a human to take deliberately, not one this function grants.

    This fails CLOSED on two conditions that used to report silent success:
    a `finding` name with no matching directory (a typo, or the underscore
    vs hyphen the package name and the CLI argument do not agree on), and a
    repository with no git history at all — a tarball export with `claim.yaml`
    and `results/` both present but no `.git` to check them against. Neither
    is "nothing to report"; both mean this check cannot vouch for anything,
    which is a failure of the guarantee, not a pass.
    """
    if not _git(repo, "rev-parse", "HEAD"):
        return [
            f"{finding}: {repo} has no git history — pre-registration cannot "
            f"be checked mechanically without commits to check it against. "
            f"This is not the same as 'nothing published yet': it means the "
            f"guarantee cannot be verified at all, which must not be "
            f"reported as success."
        ]

    base = f"findings/{finding}"
    if not (repo / base).is_dir():
        return [
            f"{finding}: no such finding — {repo / base} does not exist. "
            f"Refusing to report success for a finding this check never "
            f"actually looked at."
        ]

    claim = last_commit_touching(repo, f"{base}/claim.yaml")
    artifact = _first_artifact_commit(repo, finding)

    if artifact is None:
        # Nothing published yet. That is a normal state, not a violation: a
        # claim only has to precede raw/results when either exists.
        return []
    artifact_which, artifact_commit = artifact

    if claim is None:
        return [
            f"{finding}: {base}/{artifact_which} is committed but "
            f"claim.yaml is not"
        ]

    claim_sha, claim_when = claim.split(" ", 1)
    artifact_sha, artifact_when = artifact_commit.split(" ", 1)

    if claim_sha == artifact_sha:
        return [
            f"{finding}: claim.yaml and {artifact_which} landed in the same "
            f"commit ({claim_sha[:7]}). A prediction committed alongside its "
            f"result is not a prediction."
        ]

    order = _commit_order(repo)
    if order and claim_sha in order and artifact_sha in order:
        claim_last_touch_is_after_artifact = (
            order.index(claim_sha) > order.index(artifact_sha)
        )
    else:
        # Fallback for a history the topological listing didn't cover (should
        # not happen for commits already found by the functions above).
        claim_last_touch_is_after_artifact = claim_when > artifact_when

    if claim_last_touch_is_after_artifact:
        # The ordering verdict comes from commit history (topological order),
        # never from these dates — a committer date can be forged
        # (GIT_COMMITTER_DATE) without changing where a commit sits in
        # history. Dates are included only as recorded metadata, not as
        # evidence for the ordering claim.
        return [
            f"{finding}: {artifact_which} first appeared (commit "
            f"{artifact_sha[:7]}) before claim.yaml was last modified "
            f"(commit {claim_sha[:7]}) — claim.yaml was modified after "
            f"results existed. A claim is frozen once its finding has "
            f"raw/ or results/. Recorded committer dates, for reference "
            f"only: {artifact_which} {artifact_when}, claim {claim_when}."
        ]
    return []


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("finding")
    args = parser.parse_args(argv)

    repo = Path(__file__).resolve().parents[1]
    problems = check(repo, args.finding)
    for p in problems:
        print(f"FAIL: {p}", file=sys.stderr)
    if problems:
        return 1
    print(f"ok: {args.finding} pre-registration holds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
