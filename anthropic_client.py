#!/usr/bin/env python3
"""
Simple Anthropic client helper for Ask Shorty.

Reads ANTHROPIC_API_KEY from the environment and exposes a get_client()
function that returns a configured Anthropic client.
"""

import os
from typing import Optional


_client = None


def get_client():
    """
    Return a singleton Anthropic client instance.

    This assumes the `anthropic` Python package is installed:
      pip install anthropic

    And that ANTHROPIC_API_KEY is set in the environment.
    """
    global _client
    if _client is not None:
        return _client

    try:
        from anthropic import Anthropic  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "anthropic package not installed. Run `pip install anthropic`."
        ) from e

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set in the environment. "
            "Set it before using Shorty generation."
        )

    _client = Anthropic(api_key=api_key)
    return _client

