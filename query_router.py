#!/usr/bin/env python3
"""Heuristic query router for Ask Shorty V2."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional

RouteType = Literal[
    "entity_exact",
    "topic_lookup",
    "where_in_video",
    "cross_video_compare",
    "cause_effect",
    "channel_or_date_filtered",
    "general",
]

TopicWatchSort = Literal["recent_first", "oldest_first"]


@dataclass
class RouteResult:
    route_type: RouteType
    reason: str
    # For topic_lookup: how to order the reranked pool by watch_date (None → recent_first in V2).
    topic_watch_sort: Optional[TopicWatchSort] = None


class QueryRouter:
    _COMPARE = re.compile(
        r"\b(compare|versus|vs\.?|difference|contrast|similarit(y|ies))\b", re.I
    )
    _CAUSE = re.compile(
        r"\b(why|because|cause|effect|led to|resulted in|due to|consequence)\b", re.I
    )
    _WHERE = re.compile(
        r"\b(where in (this |the |that )?video|timestamp|what time|"
        r"at what point|which part)\b",
        re.I,
    )

    # "Find the first / earliest … watched about X" (oldest-first sort in V2).
    _FIRST_WATCHED_TOPIC = re.compile(
        r"\b(?:find|show|get|give)\s+(?:me\s+)?(?:the\s+)?(?:first|earliest)\b[^\n.?!]{0,72}\b"
        r"(?:watch(?:ed)?|saw|viewed)\s+(?:on|about|regarding|related\s+to)\b"
        r"|\b(?:what(?:'s|\s+was|\s+is)?\s+)?(?:the\s+)?(?:first|earliest)\s+"
        r"(?:video|thing)?\s*(?:i\s+)?(?:watch(?:ed)?|saw|viewed)\s+(?:on|about|regarding|related\s+to)\b"
        r"|\b(?:first|earliest)\s+(?:video|thing)\s+(?:i\s+)?(?:watch(?:ed)?|saw)\s+(?:on|about|regarding|related\s+to)\b"
        r"|\b(?:earliest|first)\s+(?:video|thing)\s+(?:about|regarding|on|related\s+to)\b",
        re.I,
    )
    # "What was the last thing I watched about X?" / "last video I watched about X"
    _LAST_WATCHED_TOPIC = re.compile(
        r"\b(?:what(?:'s|\s+is|\s+was)?\s+)?(?:the\s+)?(?:last|latest|most\s+recent)\s+"
        r"(?:thing|video|one)?\s*(?:i\s+)?(?:watch(?:ed)?|saw|viewed)\s+(?:about|regarding|on|related\s+to)\b"
        r"|\b(?:last|latest)\s+(?:thing|video)\s+(?:i\s+)?(?:watch(?:ed)?|saw)\s+(?:about|regarding|on|related\s+to)\b",
        re.I,
    )
    # "What have I watched about X?" / "What've I watched about…?" — topic, not calendar routing.
    _WATCH_HISTORY_TOPIC = re.compile(
        r"\bwhat(?:'ve|\s+(?:have|had|did))\s+i\s+watch(?:ed)?\s+(about|regarding|on|related\s+to)\b"
        r"|\bwhat\s+i\s+watch(?:ed)?\s+(about|regarding|on|related\s+to)\b",
        re.I,
    )
    # Informal / non-"what" phrasing still about watch-history topics.
    _WATCH_TOPIC_COLLOQUIAL = re.compile(
        r"\b(?:videos?|stuff|things)\s+(?:i|'ve|i've|I\s+have)\s+(?:watch(?:ed)?|seen)\s+(about|regarding)\b"
        r"|\b(?:have|did)\s+i\s+watch(?:ed)?\s+anything\s+(about|regarding)\b"
        r"|\bmy\s+watch\s+history\s+(about|regarding|for)\b",
        re.I,
    )

    _TOPIC = re.compile(
        r"^\s*(what is|what are|explain|define|how does)\b", re.I
    )

    # Explicit creator / channel scoping — not mere "watch" / "videos" wording.
    _VIDEOS_FROM_ON_BY = re.compile(
        r"\bvideos?\s+(?:from|on|by)\s+(@[^\s]+|[^\s,?.!]+(?:\s+[^\s,?.!]+){0,3})\b",
        re.I,
    )
    _CHANNEL_LITERAL = re.compile(
        # "CNN channel", "the Veritasium channel", 'channel "Foo Bar"'
        r"\b[\"']?[A-Za-z0-9][A-Za-z0-9 .\-|&+]+\s+channel[\"']?\b|"
        r"\b(?:the\s+)?[A-Za-z0-9][\w .\-|]+\s+频道\b|"  # allow future i18n
        r"\bchannel\s+[:\-]\s*[\"']?[A-Za-z0-9][^\n,?]{1,72}[\"']?\b",
        re.I,
    )
    _CREATOR_CREATOR_WORD = re.compile(
        r"\b(?:creator|youtuber|publisher)\s+[\"']?[@A-Za-z0-9]",
        re.I,
    )

    # Calendar / time filters: require concrete periods, ISO dates, or anchored weekdays —
    # not bare weekday/month words (those false-trigger on general topic queries).
    _RELATIVE_CALENDAR = re.compile(
        r"last\s+(?:week|month|year)|"
        r"this\s+(?:week|month|year)|"
        r"next\s+(?:week|month|year)|"
        r"(?:last|past|previous)\s+\d+\s+(?:day|days|week|weeks|month|months|year|years)|"
        r"\d+\s+(?:days?|weeks?|months?)\s+ago|"
        r"\b(?:today|yesterday|tomorrow)\b|"
        r"\b\d{4}-\d{2}-\d{2}\b|"
        r"\bin\s+(?:19|20)\d{2}\b",
        re.I,
    )
    _MONTH_WITH_DATE_OR_YEAR = re.compile(
        r"\b(?:january|february|march|april|june|july|august|september|october|november|december)"
        r"(?:\s+\d{1,2}(?:st|nd|rd|th)?)(?:\s*,?\s*(?:19|20)\d{2})?\b|"
        r"\b(?:january|february|march|april|june|july|august|september|october|november|december)\s+"
        r"(?:19|20)\d{2}\b|"
        r"\b(?:jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)[\.\s\-]+\d{1,2}(?:st|nd|rd|th)?"
        r"(?:[\.\s\-]+(?:19|20)\d{2})?\b",
        re.I,
    )
    _MAY_AS_MONTH = re.compile(
        r"\bmay\s+(?:(?:19|20)\d{2}|\d{1,2}(?:[\s,\-]+(?:19|20)\d{2})?)\b",
        re.I,
    )
    _WEEKDAY_CALENDAR = re.compile(
        r"\b(?:last|next|this|on|since|until|from|before|after)\s+"
        r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)s?\b",
        re.I,
    )

    _UPLOAD_PUBLISHED_DATED = re.compile(
        r"\b(?:upload(?:ed)?|publish(?:ed)?|posted)\s+(?:"
        r"on|in|between|before|after|during|around)\b",
        re.I,
    )

    _ENTITYISH = re.compile(
        r"^(who is|who was|tell me about)\s+[\"']?([^\"'?]+)[\"']?\s*\??$", re.I
    )

    def classify(self, query: str) -> RouteResult:
        q = (query or "").strip()

        if self._COMPARE.search(q):
            return RouteResult("cross_video_compare", "compare/cross-video cue")
        if self._CAUSE.search(q):
            return RouteResult("cause_effect", "causal language")
        if self._WHERE.search(q):
            return RouteResult("where_in_video", "in-video / timestamp cue")
        m = self._ENTITYISH.match(q)
        if m and len(m.group(2).split()) <= 6:
            return RouteResult("entity_exact", "short proper-like entity query")

        if self._FIRST_WATCHED_TOPIC.search(q):
            return RouteResult(
                "topic_lookup",
                "first/earliest watch + explicit topic (about/on)",
                topic_watch_sort="oldest_first",
            )
        if self._LAST_WATCHED_TOPIC.search(q):
            return RouteResult(
                "topic_lookup",
                "last/most-recent watch + explicit topic (about/on)",
                topic_watch_sort="recent_first",
            )
        # Before generic topic patterns — distinguishes history-topic from date routing.
        if self._WATCH_HISTORY_TOPIC.search(q):
            return RouteResult("topic_lookup", "watch-history topic phrase (explicit about/on)")
        if self._WATCH_TOPIC_COLLOQUIAL.search(q):
            return RouteResult("topic_lookup", "watch-history topic (informal phrasing)")

        if self._TOPIC.search(q):
            return RouteResult("topic_lookup", "definition / explainer pattern")

        if (
            self._VIDEOS_FROM_ON_BY.search(q)
            or self._CHANNEL_LITERAL.search(q)
            or self._CREATOR_CREATOR_WORD.search(q)
            or self._UPLOAD_PUBLISHED_DATED.search(q)
            or self._RELATIVE_CALENDAR.search(q)
            or self._MAY_AS_MONTH.search(q)
            or self._WEEKDAY_CALENDAR.search(q)
            or self._MONTH_WITH_DATE_OR_YEAR.search(q)
        ):
            return RouteResult(
                "channel_or_date_filtered",
                "explicit channel/creator/date/upload-time filter cues",
            )

        # quoted phrase → treat as entity/topic pin
        if re.search(r'"[^"]{2,80}"', q):
            return RouteResult("entity_exact", "quoted span")

        return RouteResult("general", "fallback")
