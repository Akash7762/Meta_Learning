"""
tests/test_phase5_llm.py
------------------------
Checks for Phase 5: LLMDesigner's offline guarantee and selection call, and
phase5_llm_select's prompts, re-asks and retries. No test reaches the network:
every API client here is a fake.
Run with:  python -m pytest tests/ -q
"""

import os
import sys
from types import SimpleNamespace
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

import phase5_llm_select as P5
from aad.llm_designer import FEATURE_DEFINITIONS, LLMDesigner

SPACE = {"algorithm": ["ea", "MPA", "EO"]}


class FakeClient:
    """Stands in for google.genai.Client; returns the given texts in turn."""

    def __init__(self, texts):
        self.texts, self.configs = list(texts), []
        self.models = self

    def generate_content(self, model, contents, config):
        self.configs.append(config)
        usage = SimpleNamespace(prompt_token_count=10, candidates_token_count=3,
                                thoughts_token_count=0, total_token_count=13)
        return SimpleNamespace(text=self.texts.pop(0), model_version="fake-1",
                               usage_metadata=usage)


def online(texts, **kw):
    d = LLMDesigner(api_key="not-a-real-key", config_space=SPACE, **kw)
    d._client = FakeClient(texts)
    return d


# --- offline guarantee -----------------------------------------------------

def test_api_key_none_stays_offline_whatever_the_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "set-in-environment")
    d = LLMDesigner(api_key=None, config_space=SPACE)
    assert d._use_llm is False

    def boom():
        raise AssertionError("a client was created")

    monkeypatch.setattr(d, "_get_client", boom)
    assert d.propose_algorithm("a smooth problem")["algorithm"] in SPACE["algorithm"]
    with pytest.raises(RuntimeError):
        d.select_algorithm("anything")


def test_from_env_needs_a_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        LLMDesigner.from_env()


# --- parsing and the selection call ----------------------------------------

@pytest.mark.parametrize("text,expected", [
    ('{"algorithm": "MPA"}', "MPA"),
    ('```json\n{"algorithm": "EO"}\n```', "EO"),
    ('{"algorithm": "CMA_ES"}', None),           # not registered
    ('{"algorithm": "mpa"}', None),              # exact names only
    ('{"choice": "MPA"}', None),
    ("MPA", None),
    ("", None),
    ('{"algorithm": ["MPA"]}', None),
])
def test_parse_selection(text, expected):
    name, error = LLMDesigner.parse_selection(text, SPACE["algorithm"])
    assert name == expected
    assert (error is None) == (expected is not None)


def test_select_algorithm_sends_fixed_settings_and_schema():
    d = online(['{"algorithm": "MPA"}'], temperature=0.0, seed=7)
    rec = d.select_algorithm("some problem")
    assert rec["algorithm"] == "MPA" and rec["valid"] and rec["error"] is None
    assert rec["model_version"] == "fake-1" and rec["tokens"]["total_token_count"] == 13
    cfg = d._client.configs[0]
    assert cfg.temperature == 0.0 and cfg.seed == 7
    assert cfg.response_mime_type == "application/json"
    assert cfg.response_schema.properties["algorithm"].enum == SPACE["algorithm"]


def test_unusable_answer_is_reported_not_replaced():
    rec = online(['{"algorithm": "CMA_ES"}']).select_algorithm("x")
    assert rec["valid"] is False and rec["algorithm"] is None
    assert "CMA_ES" in rec["error"]


# --- re-asks and retries ---------------------------------------------------

def test_run_slot_asks_again_then_gives_up():
    d = online(["not json", '{"algorithm": "x"}', '{"algorithm": "y"}'])
    slot = P5.run_slot(d, "x", P5.Pacer(0), sleep=lambda s: None)
    assert slot["pick"] is None and len(slot["attempts"]) == 1 + P5.MAX_REASK


def test_run_slot_stops_at_the_first_usable_answer():
    d = online(["not json", '{"algorithm": "EO"}', '{"algorithm": "MPA"}'])
    slot = P5.run_slot(d, "x", P5.Pacer(0), sleep=lambda s: None)
    assert slot["pick"] == "EO" and len(slot["attempts"]) == 2


class Flaky:
    def __init__(self, errors_first):
        self.errors = list(errors_first)

    def select_algorithm(self, text):
        if self.errors:
            raise self.errors.pop(0)
        return {"raw": '{"algorithm": "EO"}', "algorithm": "EO", "valid": True,
                "error": None, "model_version": None, "tokens": None}


def test_network_errors_are_retried():
    waits = []
    rec = P5.call_with_retries(Flaky([OSError("down"), TimeoutError("slow")]), "x",
                               P5.Pacer(0), sleep=waits.append)
    assert rec["algorithm"] == "EO" and rec["transport_retries"] == 2
    assert waits == list(P5.BACKOFF[:2])


def test_other_errors_stop_at_once():
    with pytest.raises(ValueError):
        P5.call_with_retries(Flaky([ValueError("bad request")]), "x", P5.Pacer(0),
                             sleep=lambda s: None)


# --- prompts built from the real matrices ----------------------------------

@pytest.fixture(scope="module")
def test_prompts():
    train, test = P5.load_matrix(P5.TRAIN_PATH), P5.load_matrix(P5.EVAL_PATH)
    designer = LLMDesigner(config_space=P5.selection_space(test))
    return test, P5.build_prompts(test, range(len(test.rows)), designer)


def test_no_function_name_in_arms_b_and_d(test_prompts):
    test, prompts = test_prompts
    assert len(prompts) == len(test.rows) * len(P5.ARMS)
    assert P5.leaks(prompts, test.funcs) == []


def test_each_arm_carries_exactly_its_information(test_prompts):
    test, prompts = test_prompts
    for i, r in enumerate(test.rows):
        for arm, show in P5.ARMS.items():
            prompt = prompts[f"{r}|{arm}"][0]
            assert (test.funcs[i] in prompt) == show["name"]
            assert ("Landscape features" in prompt) == show["features"]
            if show["features"]:
                assert all(f"- {k} = " in prompt for k in FEATURE_DEFINITIONS)
