"""Talk to a local model, and remember what it said.

WHY THE CACHE IS LOAD-BEARING
-----------------------------
This experiment runs 1440 generations. Without a cache, every re-run costs
hours and every analysis change costs them again, which in practice means the
analysis stops being re-run. With one, a re-run is free and the raw outputs
become an artifact anyone can re-analyse without a model at all.

WHAT IT REFUSES TO CACHE
------------------------
Failures and empty responses. Caching either would make a transient outage
permanent, and an empty answer is indistinguishable from a model that refused,
so it is surfaced rather than stored.

WHY WRITES ARE ATOMIC
----------------------
A cache entry is written to a temp file in the same directory and then moved
onto its final name with `os.replace`, which is atomic on both POSIX and
Windows. This is not a performance concern; it is what makes the cache safe
to treat as a source of truth. Without it, a process killed mid-write (or a
full disk) would leave a truncated `.txt` file sitting at the final path, and
`get` would serve it as a complete response with no error and no signal that
anything was wrong -- the answer extractor would then score a chain of
thought that stops mid-sentence as though it were what the model actually
said, and the corruption would be permanent and invisible. Atomic replace
means `get` only ever observes a whole file or no file; there is no third,
half-written state to accidentally trust.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path

from findings.cot_unfaithfulness.prompts import TERMINATOR

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

# WHY THESE OPTIONS, PRECISELY
# -----------------------------
# A small model does not reliably stop once it has answered the question we
# asked: given tests/fixtures/smoke_responses.json (24 real, uncapped
# completions from the three replication models), 6 of the 24 keep going
# past their own answer, either re-deriving the few-shot demonstrations from
# scratch or appending a summary that silently relabels a demonstration's
# answer as if it were the real one. Under the biased condition every
# demonstration's answer is forced to (A), so a model that wanders into one
# of these is scored as "answered A" regardless of what it actually
# concluded - manufacturing exactly the effect this replication is testing
# for, from generation length alone.
#
# A previous version of this fix used `stop` as a blocklist of the specific
# summary phrasings observed in those 24 responses ("Summary of Answers",
# "In summary", ...). That is the wrong shape of solution: re-running the
# CURRENT extractor over the same fixture found 3 more run-on responses the
# blocklist did nothing for, because they wandered into a second question
# with no header or summary phrasing at all - just different words. At 1440
# generations, models will run on in phrasings nobody sampled, and no
# finite keyword list can be complete against that.
#
# `stop`: prompts.TERMINATOR, the sentinel line every demonstration ends
# with (see prompts.py for the full reasoning and the terminator's own
# rejected alternatives). This is structural, not lexical: a model that has
# seen the pattern six times in-context emits the terminator once it has
# stated its own answer, and generation halts there - it cannot reach a
# second question or a summary, whatever words it would have used, because
# there is no length of run-on text for the stop match to wait through. The
# eight-phrase blocklist above is deliberately not kept alongside this: if
# the sentinel works, the blocklist is redundant; if it doesn't, keeping the
# blocklist would silently hide that failure behind a lexical patch instead
# of surfacing it. tests/fixtures/smoke_responses.json remains as the
# historical evidence of the pre-sentinel failure, and prompts.extract_answer
# is still hardened independently as a fallback (see its docstring) for the
# same reason a stop sequence alone was never a complete answer.
#
# `num_predict`: 2048 tokens, raised from 1024. The 1024 figure was sized
# against the same fixture's largest LEGITIMATE single-question response
# (2644 characters, ~660 tokens, gemma2:2b on disambiguation_qa, unbiased) -
# but that fixture also contains the 6 run-on responses, up to 6148
# characters (~1537 tokens) uncapped, which the old cap was silently
# competing against too: at 1024 tokens, a mid-length run-on can still finish
# without ever running out of budget, so the cap alone never was what
# stopped run-on - the blocklist was doing that work, badly. With the
# sentinel now doing the actual stopping, `num_predict` only has to cover
# legitimate reasoning about the one real question, and does not have to
# double as an emergency brake against a multi-question response. 2048
# leaves roughly 3x headroom over the 660-token legitimate maximum observed
# in this fixture, for a model that reasons more verbosely about a harder
# question than any of the 24 sampled, while still being a real bound: a
# response that runs past it and never stated an answer is truncated and
# correctly scored `None` by the extractor rather than silently misread, and
# `None` is the honest outcome for a truncated response, not a wrong one.
DEFAULT_OPTIONS = {
    "temperature": 0,
    "num_predict": 2048,
    "stop": [TERMINATOR],
}


class ModelError(Exception):
    """Raised when a model is unreachable, fails, or returns nothing."""


def cache_key(model: str, prompt: str, options: dict) -> str:
    payload = json.dumps(
        {"model": model, "prompt": prompt, "options": options},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Cache:
    """One file per response, named by its key. Boring on purpose."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.txt"

    def get(self, key: str) -> str | None:
        path = self._path(key)
        if not path.is_file():
            return None
        return path.read_bytes().decode("utf-8")

    def put(self, key: str, value: str) -> None:
        """Write atomically: stage in the same directory, then replace.

        `get` must never observe a half-written file. So the bytes go to a
        uniquely-named temp file in `self.root` first (same directory, so
        the follow-up rename is on one filesystem and therefore atomic), and
        only a successful write is moved onto the final path with
        `os.replace`. If anything goes wrong before that replace, the temp
        file is removed and the final path is left exactly as it was --
        either untouched, or still holding whatever was there before.
        """
        data = value.encode("utf-8")
        fd, tmp_name = tempfile.mkstemp(
            dir=self.root, prefix=f".{key}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            os.replace(tmp_name, self._path(key))
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise


def ask_ollama(prompt: str, model: str) -> str:
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": DEFAULT_OPTIONS,
    }).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_URL, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.loads(response.read())["response"]
    except Exception as exc:
        raise ModelError(f"{model}: {exc}") from exc


def available_models() -> dict[str, str]:
    """Model name -> content digest, from the local Ollama daemon.

    M1: this used to return just the list of names, and nothing called it.
    A name is not a stable identity -- Ollama tags are mutable, so
    `gemma2:2b` pulled a year from now is not necessarily the same weights
    as `gemma2:2b` today. The digest is; `run.manifest` records it per
    model so a later reader can tell whether a re-run actually used "the
    same model" or just the same name.
    """
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags",
                                    timeout=10) as response:
            models = json.loads(response.read())["models"]
    except Exception as exc:
        raise ModelError(f"cannot list models: {exc}") from exc
    return {m["name"]: m.get("digest", "") for m in models}


def generate(model: str, prompt: str, cache: Cache, ask=None,
             options: dict | None = None) -> str:
    """Return the model's completion, from cache when it is already known."""
    opts = DEFAULT_OPTIONS if options is None else options
    key = cache_key(model, prompt, opts)
    hit = cache.get(key)
    if hit is not None:
        return hit
    call = ask if ask is not None else ask_ollama
    text = call(prompt, model)
    if not (text or "").strip():
        raise ModelError(f"{model}: returned an empty response")
    cache.put(key, text)
    return text
