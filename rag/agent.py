"""RAG agent orchestration for CourseMate."""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

from config import (
    CHAT_MODEL,
    IMAGES_DIR,
    IMAGE_MODEL,
    OPENAI_API_BASE,
    OPENAI_API_KEY,
    PROJECT_ROOT,
    TOP_K,
    VISION_MODEL,
)
from data_pipeline.embeddings import EmbeddingClient
from data_pipeline.vector_store import HybridVectorStore
from rag import prompts
from rag.image_gen import ImageGenerator


class RAGAgent:
    def __init__(
        self,
        model: str = os.getenv("CHAT_MODEL", CHAT_MODEL),
        vision_model: str = os.getenv("VISION_MODEL", VISION_MODEL),
        image_model: str = os.getenv("IMAGE_MODEL", IMAGE_MODEL),
        api_key: str = OPENAI_API_KEY,
        base_url: str = OPENAI_API_BASE,
        top_k: int = TOP_K,
        image_generator: Optional[ImageGenerator] = None,
    ) -> None:
        self.model = model
        self.vision_model = vision_model
        self.image_model = image_model
        self.api_key = api_key
        self.base_url = base_url
        self.top_k = top_k

        # Core clients
        self.embedding_client = EmbeddingClient()
        self.vector_store = HybridVectorStore(embedding_client=self.embedding_client)
        # Increase the default timeout to 120 seconds in case the SJTU gateway is slow
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=120.0)

        # Image generator can be injected; otherwise create default wired to our retriever
        self.image_generator = image_generator or ImageGenerator(
            retriever=self.retrieve_context,
            client=self.client,
            vision_model=self.vision_model,
            image_model=self.image_model,
        )
        self.draw_triggers = getattr(prompts, "DRAW_KEYWORDS", ["generate image", "draw", "paint", "illustration"])
        self.system_prompt = getattr(prompts, "SYSTEM_PROMPT", "You are a helpful course assistant.")

    # ------------------------------------------------------------------
    def apply_runtime_settings(
        self,
        *,
        llm_api_key: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        chat_model: Optional[str] = None,
        vision_model: Optional[str] = None,
        image_model: Optional[str] = None,
        top_k: Optional[int] = None,
        embedding_provider: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> None:
        """Apply runtime settings without recreating the whole agent."""
        self.api_key = llm_api_key if llm_api_key is not None else self.api_key
        self.base_url = llm_base_url if llm_base_url is not None else self.base_url
        self.model = chat_model if chat_model is not None else self.model
        self.vision_model = vision_model if vision_model is not None else self.vision_model
        self.image_model = image_model if image_model is not None else self.image_model
        if top_k is not None:
            self.top_k = int(top_k)

        # Re-create OpenAI-compatible client so API key/base url changes take effect immediately.
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=120.0)

        # Re-bind image generator to updated models/client.
        if self.image_generator:
            self.image_generator.client = self.client
            self.image_generator.vision_model = self.vision_model
            self.image_generator.image_model = self.image_model

        # Update query embedding path for retrieval.
        if embedding_provider is not None or embedding_api_key is not None or embedding_base_url is not None or embedding_model is not None:
            provider = embedding_provider if embedding_provider is not None else self.embedding_client.provider
            api_key = embedding_api_key if embedding_api_key is not None else self.embedding_client.api_key
            base_url = embedding_base_url if embedding_base_url is not None else self.embedding_client.base_url
            model = embedding_model if embedding_model is not None else self.embedding_client.model
            self.embedding_client = EmbeddingClient(
                provider=provider,
                api_key=api_key,
                base_url=base_url,
                model=model,
                gemini_api_key=api_key,
                gemini_model=model,
            )
            self.vector_store.embedding_client = self.embedding_client

    # ------------------------------------------------------------------
    def retrieve_context(
        self, query: str, course_names: Optional[List[str]] = None, filenames: Optional[List[str]] = None, top_k: Optional[int] = None
    ) -> Tuple[str, List[Dict], List[str]]:
        """Search vector store and format context string plus image paths."""
        k = top_k or self.top_k
        results = self.vector_store.search(query, top_k=k, course_names=course_names, filenames=filenames)
        context_parts: List[str] = []
        image_paths: List[str] = []

        for idx, item in enumerate(results, 1):
            content = item.get("content", "").strip()
            meta = item.get("metadata", {})
            fname = meta.get("filename", "unknown")
            page = meta.get("page_number", 0)
            img_str = meta.get("image_paths", "")
            if img_str:
                for p in img_str.split(","):
                    p = p.strip()
                    if p:
                        image_paths.append(p)

            context_parts.append(f"--- Source {idx} [file: {fname}, page: {page}] ---\n{content}")

        context = "\n\n".join(context_parts) if context_parts else ""
        return context, results, image_paths

    # ------------------------------------------------------------------
    def _build_messages(
        self,
        query: str,
        context: str,
        chat_history: Optional[List[Dict]],
        image_paths: List[str],
        user_image_b64: Optional[str],
    ) -> List[Dict]:
        messages: List[Dict] = [{"role": "system", "content": self.system_prompt}]
        if chat_history:
            messages.extend(chat_history)

        user_content: List[Dict] = []
        body = (
            f"Use the course context below to answer the question. Always cite file and page.\n\n"
            f"--- Course Context ---\n{context or '(no context retrieved)'}\n\n"
            f"--- Question ---\n{query}"
        )
        user_content.append({"type": "text", "text": body})

        for p in image_paths[:3]:
            b64 = self._encode_image(p)
            if b64:
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

        if user_image_b64:
            user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{user_image_b64}"}})

        messages.append({"role": "user", "content": user_content})
        return messages

    # ------------------------------------------------------------------
    def answer_question_stream(
        self,
        query: str,
        chat_history: Optional[List[Dict]] = None,
        course_names: Optional[List[str]] = None,
        filenames: Optional[List[str]] = None,
        user_image_b64: Optional[str] = None,
    ):
        # Drawing intent delegated to image generator if available
        if self._is_draw_intent(query) and self.image_generator:
            return self.image_generator.generate(query, chat_history, course_names)

        context, docs, image_paths = self.retrieve_context(
            query, course_names=course_names, filenames=filenames
        )
        retrieval_error = getattr(self.vector_store, "last_search_error", None)
        if retrieval_error:
            def stream_error():
                yield f"[retrieval error] {retrieval_error}"
            return stream_error, docs

        messages = self._build_messages(query, context, chat_history, image_paths, user_image_b64)
        current_model = self.vision_model if (image_paths or user_image_b64) else self.model

        def stream():
            try:
                resp = self.client.chat.completions.create(
                    model=current_model, messages=messages, temperature=0.7, stream=True
                )
                for chunk in resp:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta
            except Exception as exc:
                yield f"[error] {exc}"

        return stream, docs

    # ------------------------------------------------------------------
    def _is_draw_intent(self, query: str) -> bool:
        lowered = query.lower()
        return any(k.lower() in lowered for k in self.draw_triggers)

    @staticmethod
    def _encode_image(path: str) -> str:
        resolved = RAGAgent._resolve_image_path(path)
        if not resolved:
            return ""
        import base64

        with open(resolved, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def _resolve_image_path(path: str) -> str:
        if not path:
            return ""
        if os.path.isabs(path) and os.path.exists(path):
            return path

        candidate = os.path.join(str(PROJECT_ROOT), path)
        if os.path.exists(candidate):
            return candidate

        legacy_candidate = os.path.join(str(IMAGES_DIR), os.path.basename(path))
        if os.path.exists(legacy_candidate):
            return legacy_candidate
        return ""
