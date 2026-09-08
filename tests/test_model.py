"""The cache is what makes this replication cheap to re-run and cheap to audit.

Every test here injects a fake model. Nothing in this suite contacts Ollama.
"""

from pathlib import Path

import pytest

from findings.cot_unfaithfulness.model import Cache, ModelError, cache_key, generate


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
