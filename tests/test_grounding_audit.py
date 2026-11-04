"""V2 grounding audit: numbers / quoted phrases in the answer that are not in the context."""

from __future__ import annotations

import pytest

import ask_shorty_v2 as v2
from tests.fakes import FakeLLM, install_fake_llm


def audit(answer: str, blob: str):
    flags = []
    v2._append_v2_grounding_audit_flags(answer, blob, flags)
    return flags


CTX = (
    "VIDEO: Async Rust on a $4 Chip with Embassy\nSHORTY:\nfirmware 61,204 bytes -> 34,899 bytes; "
    "Rust 1.75 stabilised async fn in traits on 2023-12-28; BME280 polled every 500 ms.\n"
    "Kestrel shipped on March 14, 2021. Sagan: a bottle cast into the cosmic ocean."
)


def test_clean_answer_has_no_flags():
    assert audit("Rust 1.75 shrank firmware to 34,899 bytes, polling every 500 ms.", CTX) == []


def test_novel_number_is_flagged():
    flags = audit("The firmware was 29,000 bytes and builds took 45 seconds.", CTX)
    assert flags == ["grounding_audit: answer_numbers_not_in_context=29000,45"]


def test_single_digit_numbers_ignored():
    assert audit("It took 3 tries.", CTX) == []


@pytest.mark.parametrize("ans", ["34899 bytes", "34,899 bytes", "34 899 bytes", "34 899 bytes"])
def test_thousands_separators_match_both_ways(ans):
    assert audit(f"Firmware: {ans}", CTX) == []
    assert audit("Firmware: 34,899 bytes", "firmware 34899 bytes") == []


@pytest.mark.parametrize(
    "answer",
    ["It shipped on March 14 2021.", "Polled every 500 ms in 2023 12 times."],
)
def test_denoise_does_not_glue_separate_numbers(answer):
    """Regression: whitespace between numbers was stripped: 'March 14 2021' -> '142021'."""
    flags = audit(answer, CTX + " 12 times")
    assert flags == []


def test_strip_helper_examples():
    assert v2._strip_thousands_separators("1,234,567 and 34,899") == "1234567 and 34899"
    assert v2._strip_thousands_separators("March 14, 2021 and 14 2021") == "March 14, 2021 and 14 2021"
    assert v2._strip_thousands_separators("1,23") == "1,23"


def test_quoted_phrase_not_in_context_is_flagged():
    flags = audit('Sagan called it "a message to future galactic empires".', CTX)
    assert flags == ["grounding_audit: quoted_phrase_not_in_context=a message to future galactic empires"]


def test_quoted_phrase_in_context_with_punctuation_differences():
    assert audit('Sagan: "A bottle cast into the cosmic ocean,"', CTX) == []


def test_curly_quotes_are_audited():
    """Regression: only straight quotes were checked; models usually emit “curly” ones."""
    flags = audit("Sagan called it “a message to future galactic empires”.", CTX)
    assert flags and flags[0].startswith("grounding_audit: quoted_phrase_not_in_context=")
    assert audit("Sagan said “a bottle cast into the cosmic ocean”.", CTX) == []


def test_empty_context_flag():
    assert audit("anything", "   ") == ["grounding_audit: empty_context_blob"]


def test_normalize_phrase():
    assert v2._normalize_phrase_for_match("  Hello,   World!! ") == "hello world"


def _answer_with(monkeypatch, corpus, text, question):
    install_fake_llm(monkeypatch, FakeLLM(responder=lambda s, u: text if "Ask Shorty (V2)" in s else "[]"))
    return v2.AskShortyV2(db_path=corpus["db_path"]).answer(question)


def test_e2e_flags_hallucinated_number(corpus, monkeypatch):
    res = _answer_with(monkeypatch, corpus, "Voyager carried 999 images.", "How many images are on the Voyager Golden Record?")
    assert any("answer_numbers_not_in_context=999" in f for f in res["grounding_audit"])


def test_e2e_number_from_question_is_not_evidence(corpus, monkeypatch):
    """
    Regression: the audit blob was the whole user prompt, which includes the question.
    A number the user typed (and the model echoed) passed even though no context had it.
    """
    res = _answer_with(monkeypatch, corpus, "Yes, Voyager carried 4096 images.", "Did Voyager carry 4096 images on the Golden Record?")
    assert any("4096" in f for f in res["grounding_audit"])


def test_e2e_clean_answer(corpus, fake_llm):
    res = v2.AskShortyV2(db_path=corpus["db_path"]).answer("How many images are on the Voyager Golden Record?")
    assert res["grounding_audit"] == []
    assert "115 images" in res["answer"]
