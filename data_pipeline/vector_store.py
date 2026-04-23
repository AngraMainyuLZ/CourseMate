"""Vector store abstractions for CourseMate.

Provides dense (Chroma) search plus an optional hybrid mode that fuses
dense retrieval with BM25 using reciprocal rank fusion.

Embedding generation is delegated to an injected client with a
`get_embedding(text: str) -> List[float]` method (to be implemented in
`embeddings.py`).
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings
import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from config import (
    COLLECTION_NAME,
    TELEMETRY,
    TOP_K,
    VECTOR_DB_PATH,
)


class VectorStore:
    def __init__(
        self,
        embedding_client,
        db_path: str | os.PathLike = VECTOR_DB_PATH,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self.embedding_client = embedding_client
        self.last_search_error: Optional[str] = None
        os.makedirs(db_path, exist_ok=True)

        self.chroma = chromadb.PersistentClient(
            path=str(db_path), settings=Settings(anonymized_telemetry=TELEMETRY)
        )
        self.collection = self.chroma.get_or_create_collection(
            name=collection_name, metadata={"description": "Course materials"}
        )

    # ------------------------------------------------------------------
    def add_documents(self, chunks: List[Dict[str, str]]) -> None:
        if not chunks:
            print("[vector_store] no chunks to add")
            return

        ids: List[str] = []
        embeddings: List[List[float]] = []
        documents: List[str] = []
        metadatas: List[Dict] = []

        for i, chunk in enumerate(tqdm(chunks, desc="Embedding & upsert", unit="chunk")):
            content = chunk.get("content", "")
            if not content:
                continue

            filename = chunk.get("filename", "unknown")
            page_number = chunk.get("page_number", 0)
            chunk_id = chunk.get("chunk_id", i)
            doc_id = f"{filename}-{page_number}-{chunk_id}"

            try:
                embed = self.embedding_client.get_embedding(content)
            except Exception as exc:
                print(f"[vector_store] embedding failed for {doc_id}: {exc}")
                continue

            ids.append(doc_id)
            embeddings.append(embed)
            documents.append(content)
            metadatas.append(
                {
                    "course_name": chunk.get("course_name", "default"),
                    "filename": filename,
                    "filepath": chunk.get("filepath", ""),
                    "filetype": chunk.get("filetype", ""),
                    "page_number": page_number,
                    "chunk_id": chunk_id,
                    "image_paths": chunk.get("images", ""),
                    "id": doc_id,
                }
            )

        if not ids:
            print("[vector_store] nothing embedded successfully")
            return

        self.collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        print(f"[vector_store] added {len(ids)} chunks")

    # ------------------------------------------------------------------
    def search(
        self, 
        query: str, 
        top_k: int = TOP_K, 
        course_names: Optional[List[str]] = None,
        filenames: Optional[List[str]] = None
    ) -> List[Dict]:
        self.last_search_error = None
        try:
            query_embedding = self.embedding_client.get_embedding(query)
        except Exception as exc:
            self.last_search_error = f"query embedding failed: {exc}"
            print(f"[vector_store] {self.last_search_error}")
            return []

        filters = []
        if course_names:
            if len(course_names) == 1:
                filters.append({"course_name": course_names[0]})
            else:
                filters.append({"course_name": {"$in": course_names}})
        
        if filenames:
            if len(filenames) == 1:
                filters.append({"filename": filenames[0]})
            else:
                filters.append({"filename": {"$in": filenames}})

        where_filter = None
        if len(filters) == 1:
            where_filter = filters[0]
        elif len(filters) > 1:
            where_filter = {"$and": filters}

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            self.last_search_error = f"vector query failed: {exc}"
            print(f"[vector_store] {self.last_search_error}")
            return []

        formatted: List[Dict] = []
        if not results or not results.get("documents") or not results["documents"][0]:
            return formatted

        for content, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            formatted.append({"content": content, "metadata": meta, "distance": dist})

        return formatted

    # ------------------------------------------------------------------
    def delete_by_filename(self, course_name: str, filename: str) -> None:
        """Delete all chunks from a specific file within a course."""
        where_filter = {
            "$and": [
                {"course_name": course_name},
                {"filename": filename}
            ]
        }
        self.collection.delete(where=where_filter)
        print(f"[vector_store] deleted documents with filename: {filename} in course: {course_name}")

    # ------------------------------------------------------------------
    def clear(self) -> None:
        name = self.collection.name
        self.chroma.delete_collection(name)
        self.collection = self.chroma.create_collection(name=name)
        print("[vector_store] collection cleared")

    def count(self) -> int:
        return self.collection.count()


class HybridVectorStore(VectorStore):
    """Hybrid search that fuses dense (Chroma) with BM25 sparse retrieval."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bm25 = None
        self._corpus_ids: List[str] = []
        self._corpus_tokens: List[List[str]] = []
        self._id_to_doc: Dict[str, str] = {}
        self._id_to_meta: Dict[str, Dict] = {}
        self._refresh_sparse_index()

    # ------------------------------------------------------------------
    def add_documents(self, chunks: List[Dict[str, str]]) -> None:
        super().add_documents(chunks)
        self._refresh_sparse_index()

    def search(
        self, 
        query: str, 
        top_k: int = TOP_K, 
        course_names: Optional[List[str]] = None,
        filenames: Optional[List[str]] = None
    ) -> List[Dict]:
        dense_results = super().search(
            query, 
            top_k=top_k * 2, 
            course_names=course_names,
            filenames=filenames
        )

        if not self._bm25:
            return dense_results[:top_k]

        tokenized_query = list(jieba.cut(query))
        scores = self._bm25.get_scores(tokenized_query)

        sparse_candidates: List[Dict] = []
        for idx in np.argsort(scores)[::-1]:
            if scores[idx] <= 0:
                continue
            doc_id = self._corpus_ids[idx]
            meta = self._id_to_meta.get(doc_id, {})
            
            if course_names and meta.get("course_name") not in course_names:
                continue
            if filenames and meta.get("filename") not in filenames:
                continue
                
            sparse_candidates.append(
                {
                    "id": doc_id,
                    "content": self._id_to_doc.get(doc_id, ""),
                    "metadata": meta,
                    "score": scores[idx],
                }
            )
            if len(sparse_candidates) >= top_k * 2:
                break

        return self._reciprocal_rank_fusion(dense_results, sparse_candidates, top_k=top_k)

    def delete_by_filename(self, course_name: str, filename: str) -> None:
        """Delete from Chroma and refresh sparse index."""
        super().delete_by_filename(course_name, filename)
        self._refresh_sparse_index()

    # ------------------------------------------------------------------
    def _refresh_sparse_index(self) -> None:
        data = self.collection.get(include=["documents", "metadatas"])
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []
        ids = data.get("ids") or []

        if not docs:
            self._bm25 = None
            return

        self._corpus_ids = ids
        self._id_to_doc = {doc_id: doc for doc_id, doc in zip(ids, docs)}
        self._id_to_meta = {doc_id: meta for doc_id, meta in zip(ids, metas)}
        self._corpus_tokens = [list(jieba.cut(doc)) for doc in docs]
        self._bm25 = BM25Okapi(self._corpus_tokens)
        print(f"[hybrid] BM25 index built on {len(docs)} docs")

    @staticmethod
    def _reciprocal_rank_fusion(dense_results, sparse_results, top_k: int, k: int = 60) -> List[Dict]:
        scores = {}
        doc_map = {}

        for rank, res in enumerate(dense_results):
            doc_id = res.get("metadata", {}).get("id")
            if not doc_id:
                continue
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
            doc_map[doc_id] = res

        for rank, res in enumerate(sparse_results):
            doc_id = res.get("id")
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
            if doc_id not in doc_map:
                doc_map[doc_id] = res

        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        final = []
        for doc_id, score in fused:
            datum = doc_map[doc_id]
            final.append(
                {
                    "id": doc_id,
                    "content": datum.get("content"),
                    "metadata": datum.get("metadata", {}),
                    "combined_score": score,
                }
            )
        return final
