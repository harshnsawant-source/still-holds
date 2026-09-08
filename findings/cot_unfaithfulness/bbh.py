"""Load BIG-Bench Hard tasks, strictly, and sample them the same way every time.

WHY THE FILES ARE COMMITTED
---------------------------
The corpus is part of the claim. If a re-run six months from now fetches a file
that has since changed, the numbers move for a reason nobody can see. The task
files are committed, and `fetch` exists only to obtain them the first time.

WHAT THIS REFUSES TO DO
-----------------------
Guess. An example whose options cannot be parsed, or whose target names an
option that does not exist, raises rather than being skipped. Silently dropping
a malformed example would change the denominator of every rate this experiment
reports without anyone noticing.
"""

from __future__ import annotations

import json
import random
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

BASE_URL = ("https://raw.githubusercontent.com/suzgunmirac/"
            "BIG-Bench-Hard/main/bbh")

_OPTION = re.compile(r"^\((?P<letter>[A-Z])\)\s*(?P<text>.+?)\s*$")


class BBHError(Exception):
    """Raised when a task file is missing, unparsable, or internally wrong."""


@dataclass(frozen=True)
class Example:
    question: str
    options: dict[str, str]
    target: str          # a bare letter, e.g. "A"


def fetch(task: str, dest: Path) -> Path:
    """Download one task file. Run once; the result is committed."""
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{task}.json"
    url = f"{BASE_URL}/{task}.json"
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read()
    except OSError as exc:
        raise BBHError(f"{task}: could not fetch {url} - {exc}") from exc
    path.write_bytes(payload)
    return path


def _parse(raw_input: str, raw_target: str, index: int) -> Example:
    if "Options:" not in raw_input:
        raise BBHError(
            f"example {index}: no 'Options:' block, so the choices cannot be "
            f"identified and the biasing manipulation cannot be applied"
        )
    question, _, block = raw_input.partition("Options:")
    options: dict[str, str] = {}
    for line in block.strip().splitlines():
        match = _OPTION.match(line.strip())
        if match:
            options[match.group("letter")] = match.group("text")
    if not options:
        raise BBHError(f"example {index}: 'Options:' block contained no options")

    target = raw_target.strip().strip("()").strip()
    if target not in options:
        raise BBHError(
            f"example {index}: target {target!r} names no option; "
            f"available are {sorted(options)}"
        )
    return Example(question=question.strip(), options=options, target=target)


def load(task: str, data_dir: Path) -> list[Example]:
    """Read one committed task file into parsed examples."""
    path = Path(data_dir) / f"{task}.json"
    if not path.is_file():
        raise BBHError(f"{task}: no task file at {path}. Run fetch first.")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BBHError(f"{task}: unreadable - {exc}") from exc
    examples = raw.get("examples")
    if not isinstance(examples, list) or not examples:
        raise BBHError(f"{task}: no examples in the file")
    return [
        _parse(item.get("input", ""), item.get("target", ""), i)
        for i, item in enumerate(examples)
    ]


def sample(examples: list[Example], n: int, seed: int) -> list[Example]:
    """A deterministic subset. The same seed always yields the same questions."""
    if n > len(examples):
        raise BBHError(f"asked for {n} examples but only {len(examples)} exist")
    rng = random.Random(seed)
    return rng.sample(list(examples), n)
