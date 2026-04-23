from __future__ import annotations

import os
from typing import Dict, List

import streamlit as st
from config import IMAGES_DIR, PROJECT_ROOT


def aggregate_docs(docs: List[Dict]) -> Dict[str, Dict]:
    aggregated: Dict[str, Dict] = {}
    for doc in docs or []:
        meta = doc.get("metadata", {})
        fname = meta.get("filename", "unknown")
        fpath = meta.get("filepath", "")
        page = meta.get("page_number", 0)
        preview = doc.get("content", "")[:160] + "..."

        if fname not in aggregated:
            aggregated[fname] = {"path": fpath, "pages": set(), "previews": []}
        aggregated[fname]["pages"].add(str(page))
        aggregated[fname]["previews"].append(f"p.{page}: {preview}")
    return aggregated


def render_sources(docs: List[Dict], index: int) -> None:
    if not docs:
        return
    with st.expander("📎 View sources"):
        sources = aggregate_docs(docs)
        for fname, info in sources.items():
            pages_str = ", ".join(sorted(list(info["pages"])))
            st.markdown(f"**{fname}** — pages {pages_str}")
            for preview in info["previews"]:
                st.caption(preview)


def render_images(docs: List[Dict]) -> None:
    all_paths = set()
    for doc in docs or []:
        meta = doc.get("metadata", {})
        img_str = meta.get("image_paths", "")
        if img_str:
            for p in img_str.split(","):
                p = p.strip()
                if p:
                    all_paths.add(p)

    if not all_paths:
        return

    with st.expander("🖼 Related images"):
        cols = st.columns(3)
        for idx, path in enumerate(sorted(all_paths)):
            resolved = _resolve_image_path(path)
            if resolved:
                with cols[idx % 3]:
                    st.image(resolved, caption=os.path.basename(resolved), width="stretch")


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
