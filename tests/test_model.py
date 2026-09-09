"""The cache is what makes this replication cheap to re-run and cheap to audit.

Every test here injects a fake model. Nothing in this suite contacts Ollama.
"""

import os
from pathlib import Path

import pytest

from findings.cot_unfaithfulness import model as model_module
from findings.cot_unfaithfulness.model import (
    DEFAULT_OPTIONS,
    Cache,
    ModelError,
    cache_key,
    generate,
)
from findings.cot_unfaithfulness.prompts import TERMINATOR


# --------------------------------------------------------------------------
# C1(a): generation must be bounded, or a run-on completion (see
# tests/fixtures/smoke_responses.json, the historical evidence of the
# pre-sentinel failure) never stops on its own.
# --------------------------------------------------------------------------

def test_default_options_cap_generation_length():
    assert isinstance(DEFAULT_OPTIONS.get("num_predict"), int)
    assert DEFAULT_OPTIONS["num_predict"] > 0


def test_default_options_stop_on_the_sentinel_terminator():
    # The fix is structural, not lexical: generation stops on the single
    # sentinel line every demonstration ends with (prompts.TERMINATOR), not
    # on a blocklist of phrasings sampled from past responses. See
    # model.py's DEFAULT_OPTIONS comment and prompts.py's TERMINATOR comment
    # for why a keyword blocklist was replaced rather than kept alongside
    # this.
    assert DEFAULT_OPTIONS.get("stop") == [TERMINATOR]


def test_cache_key_changes_with_every_input_that_changes_the_output():
    base = cache_key("m", "p", {"temperature": 0})
    assert base != cache_key("m2", "p", {"temperature": 0})
    assert base != cache_key("m", "p2", {"temperature": 0})
    assert base != cache_key("m", "p", {"temperature": 1})


def test_cache_key_is_stable_across_dict_ordering():
    a = cache_key("m", "p", {"temperature": 0, "seed": 1})
    b = cache_key("m", "p", {"seed": 1, "temperature": 0})
    assert a == b


def test_cache_round_trips(tmp_path: Path):
    c = Cache(tmp_path)
    assert c.get("k") is None
    c.put("k", "value")
    assert c.get("k") == "value"


def test_cache_survives_a_new_instance(tmp_path: Path):
    Cache(tmp_path).put("k", "value")
    assert Cache(tmp_path).get("k") == "value"


def test_generate_calls_the_model_once_then_serves_the_cache(tmp_path: Path):
    calls = []

    def fake(prompt: str, model: str) -> str:
        calls.append(prompt)
        return "answer"

    c = Cache(tmp_path)
    first = generate("m", "p", c, ask=fake)
    second = generate("m", "p", c, ask=fake)
    assert first == second == "answer"
    assert len(calls) == 1, "the second call must come from the cache"


def test_generate_does_not_cache_a_failure(tmp_path: Path):
    def boom(prompt: str, model: str) -> str:
        raise ModelError("provider down")

    c = Cache(tmp_path)
    with pytest.raises(ModelError):
        generate("m", "p", c, ask=boom)
    assert c.get(cache_key("m", "p", {"temperature": 0})) is None


def test_generate_does_not_cache_an_empty_response(tmp_path: Path):
    def empty(prompt: str, model: str) -> str:
        return "   "

    c = Cache(tmp_path)
    with pytest.raises(ModelError, match="empty"):
        generate("m", "p", c, ask=empty)


def test_put_interrupted_before_replace_leaves_no_file_at_final_path(
    tmp_path: Path, monkeypatch
):
    """A write that dies after staging but before the atomic replace must
    leave the final path exactly as if the write never happened -- not a
    truncated file that `get` would later serve as a complete response.
    """
    c = Cache(tmp_path)
    key = "k"

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(model_module.os, "replace", boom)

    with pytest.raises(OSError):
        c.put(key, "a value that would otherwise be written")

    assert c.get(key) is None
    assert not c._path(key).exists()


def test_put_interrupted_before_replace_leaves_no_temp_litter(
    tmp_path: Path, monkeypatch
):
    """A failed put must not leave a `.tmp` file behind for a later `get`
    (or a future put with the same random suffix) to trip over.
    """
    c = Cache(tmp_path)

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(model_module.os, "replace", boom)

    with pytest.raises(OSError):
        c.put("k", "value")

    assert list(tmp_path.iterdir()) == []


def test_put_get_round_trip_is_byte_identical_after_the_atomic_rewrite(
    tmp_path: Path,
):
    c = Cache(tmp_path)
    value = "La réponse est: 42 (答え) — done.   \n\n"
    c.put("k", value)
    assert c.get("k") == value


def test_put_over_existing_key_replaces_atomically_with_no_partial_mixture(
    tmp_path: Path,
):
    c = Cache(tmp_path)
    c.put("k", "old value, considerably longer than the new one")
    c.put("k", "new")
    assert c.get("k") == "new"
    # Only the final file should remain -- no temp files left beside it,
    # and no trace of the old, longer value mixed into what's on disk.
    assert [p.name for p in tmp_path.iterdir()] == ["k.txt"]
