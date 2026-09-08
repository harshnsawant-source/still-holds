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

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_OPTIONS = {"temperature": 0}


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


def available_models() -> list[str]:
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags",
                                    timeout=10) as response:
            return [m["name"] for m in json.loads(response.read())["models"]]
    except Exception as exc:
        raise ModelError(f"cannot list models: {exc}") from exc


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
