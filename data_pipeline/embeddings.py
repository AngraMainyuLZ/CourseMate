"""Embedding client abstraction for CourseMate."""
from __future__ import annotations

from typing import List

from config import (
    EMBEDDING_API_BASE,
    EMBEDDING_API_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    GEMINI_EMBEDDING_API_KEY,
    GEMINI_EMBEDDING_MODEL,
)


class EmbeddingClient:
    """Embedding client supporting both generic OpenAI-compatible and Gemini APIs."""

    def __init__(
        self,
        provider: str = EMBEDDING_PROVIDER,
        api_key: str = EMBEDDING_API_KEY,
        base_url: str = EMBEDDING_API_BASE,
        model: str = EMBEDDING_MODEL,
        gemini_api_key: str = GEMINI_EMBEDDING_API_KEY,
        gemini_model: str = GEMINI_EMBEDDING_MODEL,
    ) -> None:
        self.provider = (provider or "custom").strip().lower()
        self.api_key = api_key
        self.base_url = base_url
        self.gemini_api_key = gemini_api_key
        self.gemini_model = gemini_model

        if self.provider == "gemini":
            # Delay import so custom mode does not require google-genai at runtime.
            from google import genai

            self.model = gemini_model
            self.client = genai.Client(api_key=self.gemini_api_key)
            return

        # Default path: generic OpenAI-compatible embeddings API.
        from openai import OpenAI

        self.provider = "custom"
        self.model = model
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def get_embedding(self, text: str) -> List[float]:
        """Return embedding vector for the given text."""
        if self.provider == "gemini":
            # For gemini-embedding-2-preview, a light prefix improves doc embeddings.
            formatted_text = f"title: none | text: {text}" if "embedding-2" in self.model else text
            resp = self.client.models.embed_content(
                model=self.model,
                contents=formatted_text,
            )
            return resp.embeddings[0].values

        resp = self.client.embeddings.create(model=self.model, input=text)
        return resp.data[0].embedding
