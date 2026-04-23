"""Text splitting utilities for CourseMate RAG pipeline."""
from __future__ import annotations

import re
from typing import Dict, List

from tqdm import tqdm


class TextSplitter:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Include common Chinese and English sentence boundaries plus blank lines.
        self.sentence_separators = re.compile(r"[。？！?!\n\n]")

    # ------------------------------------------------------------------
    def split_text(self, text: str) -> List[str]:
        """Greedy sliding-window splitter with sentence-aware boundaries."""
        if not text:
            return []

        chunks: List[str] = []
        idx = 0
        n = len(text)

        while idx < n:
            target_end = min(idx + self.chunk_size, n)
            best_end = target_end

            search_start = max(idx, target_end - self.chunk_overlap)
            for match in self.sentence_separators.finditer(text, search_start, target_end):
                best_end = match.end()

            chunk = text[idx:best_end].strip()
            if chunk:
                chunks.append(chunk)

            next_start = best_end - self.chunk_overlap
            if next_start <= idx:
                next_start = best_end
            idx = next_start

        return chunks

    # ------------------------------------------------------------------
    def split_documents(self, documents: List[Dict]) -> List[Dict]:
        """Split loaded documents into embedding-ready chunks."""
        chunked: List[Dict] = []

        for doc in tqdm(documents, desc="Splitting documents", unit="doc"):
            content = doc.get("content", "")
            filetype = doc.get("filetype", "")
            images = doc.get("images", [])
            base_meta = {
                "filename": doc.get("filename", "unknown"),
                "filepath": doc.get("filepath", ""),
                "filetype": filetype,
                "page_number": doc.get("page_number", 0),
                "course_name": doc.get("course_name", "default"),
                "images": ",".join(images) if images else "",
            }

            # Page/slide chunks are already scoped—keep as-is.
            if filetype in {".pdf", ".pptx"}:
                chunked.append(
                    {
                        "content": content,
                        "chunk_id": 0,
                        **base_meta,
                    }
                )
                continue

            # Doc-level text needs further splitting.
            for i, chunk in enumerate(self.split_text(content)):
                chunked.append(
                    {
                        "content": chunk,
                        "chunk_id": i,
                        **base_meta,
                    }
                )

        print(f"\nSplit complete: {len(chunked)} chunks")
        return chunked
