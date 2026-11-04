"""
Fake LLM clients for the test suite.

Nothing here talks to the network. The fakes stand in at the OpenRouter / OpenAI
SDK boundary (``client.chat.completions.create``) and the Anthropic boundary
(``client.messages.create``) and return canned but realistic responses.

The answer fake is retrieval-honest: it reads the context the pipeline built,
picks the fixture video whose block best matches the question, and answers with
that video's canned answer. If retrieval put the wrong video in context, the
answer cites the wrong video, so tests that check citations really test retrieval.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

from tests.fixtures.corpus import VIDEOS, video_by_id

# One canned answer per fixture video. Numbers here all appear in the fixture
# context, so a clean grounding audit is expected for them.
CANNED_ANSWERS: Dict[str, str] = {
    "KSTRL42rel1": (
        "In Kestrel DB 4.2, batching WAL fsync calls cut p99 write latency from 48 ms "
        "to 11 ms, according to lead maintainer Priya Raman."
    ),
    "KSTRLpm0002": (
        "The Tidewater page-split race was fixed by contributor Marcus Oyelaran, who added "
        "a latch around the split path; the fix shipped in Kestrel 4.2.3."
    ),
    "SRDGH780003": (
        "At 78% hydration the extra water boosts amylase enzyme activity, which produces "
        "more sugar and gas, and the wet dough stretches around the bubbles for an open crumb."
    ),
    "VGRrec00004": (
        "The Voyager Golden Record holds 115 images, plus greetings in 55 languages."
    ),
    "CRRNGTN1859": (
        "The coronal mass ejection swung Earth's magnetic field, which induced currents in "
        "telegraph wires strong enough to throw sparks from the keys."
    ),
    "EMBSSYrs005": (
        "Rust 1.75 stabilised async fn in traits, and the Embassy port shrank the firmware "
        "to 34,899 bytes."
    ),
    "WGGLdnc0007": (
        "Karl von Frisch decoded the waggle dance; about one second of waggle means roughly 750 metres."
    ),
}

_STOP = {
    "the", "a", "an", "of", "in", "on", "to", "and", "is", "was", "what", "who", "why",
    "how", "did", "does", "do", "many", "much", "when", "which", "for", "get", "are",
}


def _tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if t not in _STOP and len(t) > 1]


def _context_blocks_by_video(user_prompt: str) -> Dict[str, str]:
    """Split a pipeline prompt into {video_id: text of all blocks for that video}."""
    out: Dict[str, str] = {}
    # V2 blocks are separated by '\n---\n' and carry 'VIDEO_ID: x'; V1 headers carry 'video_id=x'.
    for block in re.split(r"\n---\n", user_prompt):
        m = re.search(r"VIDEO_ID:\s*(\S+)", block) or re.search(r"video_id=(\S+)", block)
        if not m:
            continue
        vid = m.group(1)
        out[vid] = out.get(vid, "") + "\n" + block
    return out


def _question_from_prompt(user_prompt: str) -> str:
    m = re.search(r"(?:User question|Question):\s*\n(.+?)\n", user_prompt)
    return m.group(1).strip() if m else ""


def best_video_for(question: str, user_prompt: str) -> Optional[str]:
    blocks = _context_blocks_by_video(user_prompt)
    if not blocks:
        return None
    q = set(_tokens(question))
    scored = sorted(
        ((len(q & set(_tokens(txt))), -i, vid) for i, (vid, txt) in enumerate(blocks.items())),
        reverse=True,
    )
    return scored[0][2]


def canned_answer(system_prompt: str, user_prompt: str) -> str:
    question = _question_from_prompt(user_prompt)
    vid = best_video_for(question, user_prompt)
    if vid is None or vid not in CANNED_ANSWERS:
        return "I could not find this in the provided context."
    v = video_by_id(vid)
    watched = (v["watch_date"] or "")[:10]
    return (
        f"{CANNED_ANSWERS[vid]}\n\nSource: \"{v['title']}\" by {v['channel']} "
        f"(watched: {watched}) [video_id={vid}]"
    )


def default_responder(system_prompt: str, user_prompt: str) -> str:
    """Route by the prompt the project sends; return what a well-behaved model would."""
    s = system_prompt or ""
    if "query rewriting engine" in s:
        q = _question_from_prompt(user_prompt + "\n") or user_prompt.strip().splitlines()[-1]
        kw = " ".join(_tokens(q))
        return json.dumps([q.rstrip("?") + " explained", kw, "details about " + kw])
    if "JSON array of strings" in s and "alternative phrasings" in user_prompt:
        q = user_prompt.split("Original:", 1)[-1].strip()
        return json.dumps([" ".join(_tokens(q)), q.lower().rstrip("?")])
    if "metadata" in s.lower() and ("channel" in s.lower() or "date" in s.lower()) and "Ask Shorty" not in s:
        return "{}"
    if "compression engine" in s:
        return (
            "CONTEXT\nFake Shorty body produced by the test double.\n\n"
            "TOPICS\nTOPIC 1 – Test topic\n- What it is: placeholder\n\n"
            "MICRO-DETAILS\nNumbers: 1, 2, 3\n"
        )
    if "likely user questions" in s:
        return json.dumps(["What is the test topic?", "Who is involved?"])
    if "Ask Shorty" in s:
        return canned_answer(s, user_prompt)
    return "[]"


@dataclass
class FakeLLM:
    """
    Records every call. ``mode`` selects a failure to inject:
      ok | timeout | error | garbage | empty
    """

    mode: str = "ok"
    responder: Callable[[str, str], str] = default_responder
    calls: List[Dict[str, Any]] = field(default_factory=list)

    def respond(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append({"system": system_prompt, "user": user_prompt})
        if self.mode == "timeout":
            import httpx
            import openai

            raise openai.APITimeoutError(request=httpx.Request("POST", "https://openrouter.invalid/api/v1"))
        if self.mode == "error":
            raise RuntimeError("fake upstream error: 400 bad request")
        if self.mode == "garbage":
            return '{"oops": [unterminated, not json at all <<<'
        if self.mode == "empty":
            return ""
        return self.responder(system_prompt, user_prompt)

    # --- OpenAI / OpenRouter SDK surface -------------------------------------------------
    @property
    def chat(self):  # client.chat.completions.create(...)
        return SimpleNamespace(completions=SimpleNamespace(create=self._openai_create))

    def _openai_create(self, *, model: str, messages: List[Dict[str, str]], **_: Any):
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        user = "\n".join(m["content"] for m in messages if m["role"] == "user")
        text = self.respond(system, user)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])

    # --- Anthropic SDK surface -------------------------------------------------------------
    @property
    def messages(self):
        return SimpleNamespace(create=self._anthropic_create)

    def _anthropic_create(self, *, model: str, system: str = "", messages=None, tools=None, **_: Any):
        user = "\n".join(str(m.get("content")) for m in (messages or []) if m.get("role") == "user")
        text = self.respond(system, user)
        if tools:
            # Tool-use path: wrap a JSON array result in the requested tool's input.
            name = tools[0]["name"]
            try:
                arr = json.loads(text)
            except Exception:
                arr = None
            key = list(tools[0]["input_schema"]["properties"].keys())[0]
            if isinstance(arr, list):
                return SimpleNamespace(content=[{"type": "tool_use", "name": name, "input": {key: arr}}])
            return SimpleNamespace(content=[{"type": "text", "text": text}])
        return SimpleNamespace(content=[{"type": "text", "text": text}])


def install_fake_llm(monkeypatch, fake: Optional[FakeLLM] = None) -> FakeLLM:
    """Point every LLM boundary in the project at ``fake``."""
    import ask_shorty
    import anthropic_client

    fake = fake or FakeLLM()
    # Answer/rewrite path: an OpenRouter-style model id routes through _get_openrouter_client.
    monkeypatch.setattr(ask_shorty, "ANSWER_MODEL", "fake-vendor/fake-model")
    monkeypatch.setattr(ask_shorty, "_get_openrouter_client", lambda: fake)
    monkeypatch.setattr(anthropic_client, "_client", fake)
    # shorty_generator builds OpenAI(...) inline on the openrouter path.
    import openai

    monkeypatch.setattr(openai, "OpenAI", lambda *a, **k: fake)
    return fake


__all__ = ["FakeLLM", "install_fake_llm", "CANNED_ANSWERS", "VIDEOS"]
