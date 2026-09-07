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


def check(repo: Path, finding: str) -> list[str]:
    """Return the pre-registration problems for one finding. Empty means fine."""
    base = f"findings/{finding}"
    claim = first_commit_date(repo, f"{base}/claim.yaml")
    results = first_commit_date(repo, f"{base}/results")

    if results is None:
        # Nothing published yet. That is a normal state, not a violation: a
        # claim only has to precede results when results exist.
        return []

    if claim is None:
        return [f"{finding}: results are committed but claim.yaml is not"]

    claim_sha, claim_when = claim.split(" ", 1)
    results_sha, results_when = results.split(" ", 1)

    if claim_sha == results_sha:
        return [
            f"{finding}: claim.yaml and results landed in the same commit "
            f"({claim_sha[:7]}). A prediction committed alongside its result "
            f"is not a prediction."
        ]

    order = _commit_order(repo)
    if order and claim_sha in order and results_sha in order:
        results_before_claim = order.index(results_sha) < order.index(claim_sha)
    else:
        # Fallback for a history the topological listing didn't cover (should
        # not happen for commits first_commit_date already found on HEAD).
        results_before_claim = claim_when > results_when

    if results_before_claim:
        return [
            f"{finding}: results were committed before claim.yaml "
            f"({results_when} before {claim_when})."
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
