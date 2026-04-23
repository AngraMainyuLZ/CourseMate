"""Central configuration for CourseMate.

Note: API keys should be provided via environment variables, not hardcoded.
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = PROJECT_ROOT / "extracted_images"
VECTOR_DB_PATH = PROJECT_ROOT / "vector_db"
COLLECTION_NAME = "course_materials"

# ---------------------------------------------------------------------------
# Embeddings / models
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
CHAT_MODEL = os.getenv("CHAT_MODEL", "deepseek-chat")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen3vl")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "dall-e-3")

# Embedding provider selection:
# - "custom": use OpenAI-compatible embeddings endpoint (default)
# - "gemini": use google-genai Gemini embeddings endpoint
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "custom").strip().lower()

# Custom (OpenAI-compatible) embeddings settings
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", OPENAI_API_KEY)
EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE", OPENAI_API_BASE)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Gemini embeddings settings (used only when EMBEDDING_PROVIDER=gemini)
GEMINI_EMBEDDING_API_KEY = os.getenv("GEMINI_EMBEDDING_API_KEY", EMBEDDING_API_KEY)
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2-preview")

# ---------------------------------------------------------------------------
# Chunking / retrieval
# ---------------------------------------------------------------------------
CHUNK_SIZE = 512
# overlap in characters
CHUNK_OVERLAP = 64
TOP_K = 4

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
TELEMETRY = False  # disable Chroma anonymized telemetry by default
SESSIONS_DIR = PROJECT_ROOT / "sessions"
