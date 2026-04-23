"""Build the vector index for CourseMate.

Usage:
    python process_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION_NAME, DATA_DIR, IMAGES_DIR, VECTOR_DB_PATH
from core.settings import load_settings
from data_pipeline.embeddings import EmbeddingClient
from data_pipeline.loader import DocumentLoader
from data_pipeline.splitter import TextSplitter
from data_pipeline.vector_store import HybridVectorStore
from data_pipeline.tracker import FileTracker

def main() -> None:
    settings = load_settings()
    print(f"[process] data_dir={DATA_DIR}")
    
    tracker = FileTracker(str(DATA_DIR / "metadata.json"))
    loader = DocumentLoader(data_dir=str(DATA_DIR), images_dir=str(IMAGES_DIR), extract_images=True)
    splitter = TextSplitter(
        chunk_size=settings.chunk_size or CHUNK_SIZE,
        chunk_overlap=settings.chunk_overlap if settings.chunk_overlap is not None else CHUNK_OVERLAP,
    )
    embedder = EmbeddingClient(
        provider=settings.embedding_provider,
        api_key=settings.embedding_api_key,
        base_url=settings.embedding_base_url,
        model=settings.embedding_model,
        gemini_api_key=settings.embedding_api_key,
        gemini_model=settings.embedding_model,
    )
    store = HybridVectorStore(embedding_client=embedder, db_path=str(VECTOR_DB_PATH), collection_name=COLLECTION_NAME)

    docs = loader.load_all_documents()
    if not docs:
        print("[process] no documents found; aborting.")
        return

    # Filter documents based on hash
    changed_docs = []
    # Identify which files have changed
    for doc in docs:
        file_path = doc.get("filepath", "")
        # extract file properties from doc dict
        course_name = doc.get("course_name", "default")
        filename = doc.get("filename", "")
        
        if tracker.has_changed(file_path, course_name):
            # If changed, remove old entries from DB first
            print(f"[process] update detected for {course_name}/{filename}")
            store.delete_by_filename(course_name, filename)
            changed_docs.append(doc)
            
    if not changed_docs:
        print("[process] all files are up to date! skipping index rebuild.")
        return

    print(f"[process] loaded {len(changed_docs)} document chunks before splitting")
    chunks = splitter.split_documents(changed_docs)
    print(f"[process] embedding {len(chunks)} chunks ...")
    store.add_documents(chunks)
    
    # Mark as processed
    for doc in changed_docs:
        file_path = doc.get("filepath", "")
        course_name = doc.get("course_name", "default")
        tracker.mark_processed(file_path, course_name)
        
    print(f"[process] done. collection size = {store.count()}")

if __name__ == "__main__":
    main()
