"""Central configuration for the Streamlit Cloud build.

LLM credentials are read from Streamlit secrets (``st.secrets``) with a
fallback to environment variables so that the module is usable both on
Streamlit Cloud and in a plain Python environment.
"""
from __future__ import annotations

import os
from typing import Any


_SECRETS: dict = {}
try:
    import streamlit as st

    try:
        _SECRETS = dict(st.secrets)
    except Exception:
        _SECRETS = {}
except ImportError:
    pass


def _secret(key: str, default: Any = "") -> Any:
    if key in _SECRETS:
        return _SECRETS[key]
    return os.environ.get(key, default)


DEFAULT_BANK = "Odeabank"

LLM_API_KEY = str(_secret("LLM_API_KEY", ""))
LLM_BASE_URL = str(_secret("LLM_BASE_URL", ""))
LLM_MODEL = str(_secret("LLM_MODEL", ""))

LLM_TEMPERATURE = float(_secret("LLM_TEMPERATURE", 0.3))
LLM_MAX_TOKENS = int(_secret("LLM_MAX_TOKENS", 900))
LLM_TIMEOUT_S = int(_secret("LLM_TIMEOUT_S", 60))

ANOMALY_CONTAMINATION = 0.08
TOPIC_MIN_POSTS = 3
INFLUENCER_FOLLOWER_THRESHOLD = 10_000


def llm_configured() -> bool:
    return all(str(v).strip() for v in (LLM_API_KEY, LLM_BASE_URL, LLM_MODEL))
