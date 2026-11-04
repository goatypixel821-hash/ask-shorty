"""V1 query rewriting and its failure modes."""

from __future__ import annotations

import pytest

from tests.fakes import FakeLLM, install_fake_llm


@pytest.fixture()
def v1(corpus, monkeypatch):
    import ask_shorty

    monkeypatch.setenv("ASK_SHORTY_DB_PATH", corpus["db_path"])
    eng = ask_shorty.AskShorty()
    eng._rag = corpus["rag"]
    return eng


def test_rewrites_include_original_first(v1, fake_llm):
    out = v1._rewrite_query("How many images are on the Voyager Golden Record?")
    assert out[0] == "How many images are on the Voyager Golden Record?"
    assert 2 <= len(out) <= 4


@pytest.mark.parametrize("mode", ["garbage", "empty"])
def test_non_json_rewrite_does_not_search_the_prompt_template(v1, monkeypatch, mode):
    """Regression: the fallback returned the rewrite prompt template as a query variant."""
    install_fake_llm(monkeypatch, FakeLLM(mode=mode))
    out = v1._rewrite_query("Who fixed the Tidewater bug?")
    assert out == ["Who fixed the Tidewater bug?"]
    assert not any("Generate 4 search queries" in q for q in out)


@pytest.mark.parametrize("mode", ["timeout", "error"])
def test_rewrite_llm_failure_falls_back_to_question(v1, monkeypatch, mode):
    """Regression: a timeout in the optional rewrite step aborted the whole answer."""
    install_fake_llm(monkeypatch, FakeLLM(mode=mode))
    assert v1._rewrite_query("Who fixed the Tidewater bug?") == ["Who fixed the Tidewater bug?"]


def test_answer_llm_timeout_propagates(v1, monkeypatch):
    """The final answer call failing must raise (so the job is marked error, not completed)."""
    import openai

    install_fake_llm(monkeypatch, FakeLLM(mode="timeout"))
    with pytest.raises(openai.APITimeoutError):
        v1.answer_question("Who fixed the Tidewater bug?")
