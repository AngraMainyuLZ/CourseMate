"""Image generation helper for CourseMate."""
from __future__ import annotations

import base64
import os
from typing import Callable, Dict, List, Optional, Tuple

from openai import OpenAI

from config import OPENAI_API_BASE, OPENAI_API_KEY
from rag import prompts


class ImageGenerator:
    """Optimizes a drawing prompt using vision model + context, then calls image API."""

    def __init__(
        self,
        retriever: Callable[[str, Optional[List[str]], Optional[int]], Tuple[str, List[Dict], List[str]]],
        client: Optional[OpenAI] = None,
        vision_model: str = os.getenv("VISION_MODEL", "qwen-vl-max"),
        image_model: str = os.getenv("IMAGE_MODEL", "dall-e-3"),
    ) -> None:
        self.retriever = retriever
        self.vision_model = vision_model
        self.image_model = image_model
        self.client = client or OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

    # ------------------------------------------------------------------
    def generate(
        self, query: str, chat_history: Optional[List[Dict]] = None, course_names: Optional[List[str]] = None
    ):
        context, docs, image_paths = self.retriever(query, course_names=course_names)
        optimized_prompt = self._optimize_prompt(query, context, image_paths)

        def stream():
            yield "🧠 Crafting image prompt from course materials...\n\n"
            yield f"**Planned prompt**:\n{optimized_prompt}\n\n"
            try:
                img_resp = self.client.images.generate(model=self.image_model, prompt=optimized_prompt, n=1)
                url = img_resp.data[0].url
                yield f"![generated image]({url})"
            except Exception as exc:
                yield f"[image generation failed] {exc}"

        return stream, docs

    # ------------------------------------------------------------------
    def _optimize_prompt(self, query: str, context: str, image_paths: List[str]) -> str:
        """Use the vision model to synthesize a high-quality drawing prompt."""
        user_content: List[Dict] = [{"type": "text", "text": prompts.DRAW_PROMPT_TEMPLATE.format(query=query, context=context)}]

        for p in image_paths[:3]:
            b64 = self._encode_image(p)
            if b64:
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

        messages = [
            {"role": "system", "content": prompts.DRAW_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        try:
            resp = self.client.chat.completions.create(model=self.vision_model, messages=messages, max_tokens=400)
            return resp.choices[0].message.content
        except Exception:
            return f"Create a detailed illustration for: {query}"

    @staticmethod
    def _encode_image(path: str) -> str:
        if not path or not os.path.exists(path):
            return ""
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
