"""Runtime settings persistence for CourseMate.

This module is intentionally UI-agnostic so both Streamlit UI and scripts can
share one settings source of truth.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

from config import (
    BASE_DIR,
    CHAT_MODEL,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_API_BASE,
    EMBEDDING_API_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    IMAGE_MODEL,
    TOP_K,
    VISION_MODEL,
    OPENAI_API_BASE,
    OPENAI_API_KEY,
)


SETTINGS_DIR = BASE_DIR / "settings"
SETTINGS_PATH = SETTINGS_DIR / "app_settings.json"


@dataclass
class AppSettings:
    # LLM API settings (chat/vision/image generation)
    llm_provider: str = "custom"
    llm_api_key: str = OPENAI_API_KEY
    llm_base_url: str = OPENAI_API_BASE
    chat_model: str = CHAT_MODEL
    vision_model: str = VISION_MODEL
    image_model: str = IMAGE_MODEL

    # API/provider settings
    embedding_provider: str = EMBEDDING_PROVIDER
    embedding_api_key: str = EMBEDDING_API_KEY
    embedding_base_url: str = EMBEDDING_API_BASE
    embedding_model: str = EMBEDDING_MODEL

    # Runtime retrieval/settings
    top_k: int = TOP_K
    history_limit: int = 5
    context_history_messages: int = 7
    quiz_question_count: int = 5

    # Ingestion settings
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP


def _sanitize_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        iv = int(value)
    except (TypeError, ValueError):
        return default
    if iv < minimum:
        return minimum
    if iv > maximum:
        return maximum
    return iv


def normalize_settings(raw: Dict[str, Any]) -> AppSettings:
    defaults = AppSettings()
    llm_provider = str(raw.get("llm_provider", defaults.llm_provider)).strip().lower() or "custom"
    if llm_provider not in {"custom", "gemini"}:
        llm_provider = "custom"

    provider = str(raw.get("embedding_provider", defaults.embedding_provider)).strip().lower() or "custom"
    if provider not in {"custom", "gemini"}:
        provider = "custom"

    return AppSettings(
        llm_provider=llm_provider,
        llm_api_key=str(raw.get("llm_api_key", defaults.llm_api_key)),
        llm_base_url=str(raw.get("llm_base_url", defaults.llm_base_url)),
        chat_model=str(raw.get("chat_model", defaults.chat_model)),
        vision_model=str(raw.get("vision_model", defaults.vision_model)),
        image_model=str(raw.get("image_model", defaults.image_model)),
        embedding_provider=provider,
        embedding_api_key=str(raw.get("embedding_api_key", defaults.embedding_api_key)),
        embedding_base_url=str(raw.get("embedding_base_url", defaults.embedding_base_url)),
        embedding_model=str(raw.get("embedding_model", defaults.embedding_model)),
        top_k=_sanitize_int(raw.get("top_k"), defaults.top_k, 1, 50),
        history_limit=_sanitize_int(raw.get("history_limit"), defaults.history_limit, 1, 200),
        context_history_messages=_sanitize_int(raw.get("context_history_messages"), defaults.context_history_messages, 1, 40),
        quiz_question_count=_sanitize_int(raw.get("quiz_question_count"), defaults.quiz_question_count, 1, 20),
        chunk_size=_sanitize_int(raw.get("chunk_size"), defaults.chunk_size, 64, 8192),
        chunk_overlap=_sanitize_int(raw.get("chunk_overlap"), defaults.chunk_overlap, 0, 4096),
    )


def load_settings(path: Path = SETTINGS_PATH) -> AppSettings:
    if not path.exists():
        return AppSettings()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return AppSettings()
    return normalize_settings(data if isinstance(data, dict) else {})


def save_settings(settings: AppSettings, path: Path = SETTINGS_PATH) -> None:
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(settings), ensure_ascii=False, indent=2), encoding="utf-8")
