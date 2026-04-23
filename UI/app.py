"""Streamlit UI for CourseMate."""
from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import streamlit as st

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DATA_DIR, SESSIONS_DIR, VECTOR_DB_PATH, CHUNK_OVERLAP, CHUNK_SIZE, IMAGES_DIR
from core.settings import AppSettings, load_settings, normalize_settings, save_settings
from rag.agent import RAGAgent
from data_pipeline.tracker import FileTracker
from data_pipeline.loader import DocumentLoader
from data_pipeline.splitter import TextSplitter
from UI import render

SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _init_session_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("retrieved_history", [])
    st.session_state.setdefault("session_id", None)
    st.session_state.setdefault("session_title", "New Chat")


def save_session() -> None:
    if not st.session_state.messages:
        return
    if not st.session_state.session_id:
        st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:4]
    
    title = st.session_state.session_title
    if title == "New Chat" and st.session_state.messages:
        first_user_msg = next((m["content"] for m in st.session_state.messages if m["role"] == "user"), "Chat")
        title = first_user_msg[:20] + ("..." if len(first_user_msg) > 20 else "")
        st.session_state.session_title = title

    filepath = SESSIONS_DIR / f"{st.session_state.session_id}.json"
    data = {
        "id": st.session_state.session_id,
        "title": title,
        "updated_at": datetime.now().isoformat(),
        "messages": st.session_state.messages,
        "retrieved_history": st.session_state.retrieved_history,
    }
    filepath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_session(session_id: str) -> None:
    filepath = SESSIONS_DIR / f"{session_id}.json"
    if filepath.exists():
        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
            st.session_state.session_id = data.get("id", session_id)
            st.session_state.session_title = data.get("title", "Chat")
            st.session_state.messages = data.get("messages", [])
            st.session_state.retrieved_history = data.get("retrieved_history", [])
        except Exception as e:
            st.error(f"Failed to load session: {e}")


@st.cache_resource
def load_agent() -> RAGAgent:
    return RAGAgent()


def _get_settings() -> AppSettings:
    if "app_settings" not in st.session_state:
        st.session_state.app_settings = asdict(load_settings())
    return AppSettings(**st.session_state.app_settings)


def _apply_settings_to_agent(agent: RAGAgent, settings: AppSettings) -> None:
    agent.top_k = int(settings.top_k)
    agent.apply_runtime_settings(
        llm_api_key=settings.llm_api_key,
        llm_base_url=settings.llm_base_url,
        chat_model=settings.chat_model,
        vision_model=settings.vision_model,
        embedding_provider=settings.embedding_provider,
        embedding_api_key=settings.embedding_api_key,
        embedding_base_url=settings.embedding_base_url,
        embedding_model=settings.embedding_model,
        image_model=settings.image_model,
    )


def _rebuild_index_in_app(agent: RAGAgent) -> Tuple[bool, str]:
    try:
        settings = _get_settings()
        tracker = FileTracker(str(DATA_DIR / "metadata.json"))
        loader = DocumentLoader(data_dir=str(DATA_DIR), images_dir=str(IMAGES_DIR), extract_images=True)
        splitter = TextSplitter(
            chunk_size=settings.chunk_size or CHUNK_SIZE,
            chunk_overlap=settings.chunk_overlap if settings.chunk_overlap is not None else CHUNK_OVERLAP,
        )
        store = agent.vector_store
        
        docs = loader.load_all_documents()
        changed_docs = []
        for doc in docs:
            file_path = doc.get("filepath", "")
            course_name = doc.get("course_name", "default")
            filename = doc.get("filename", "")
            
            if tracker.has_changed(file_path, course_name):
                print(f"[UI Rebuild] updating {course_name}/{filename}", flush=True)
                store.delete_by_filename(course_name, filename)
                changed_docs.append(doc)
                
        if changed_docs:
            chunks = splitter.split_documents(changed_docs)
            print(f"[UI Rebuild] embedding {len(chunks)} chunks ...", flush=True)
            store.add_documents(chunks)
            for doc in changed_docs:
                file_path = doc.get("filepath", "")
                course_name = doc.get("course_name", "default")
                tracker.mark_processed(file_path, course_name)
                
        return True, f"成功更新了 {len(changed_docs)} 个变动文件"
    except Exception as e:
        print(f"[UI Rebuild] Error: {e}", flush=True)
        return False, str(e)


# ---------------------- State Persistence Helpers ----------------------

def get_static_filter_state(key: str, default_val: bool = True) -> bool:
    """Reads persistent boolean state for UI filters from DATA_DIR."""
    state_file = DATA_DIR / ".filter_state.json"
    if "filter_state" not in st.session_state:
        if state_file.exists():
            try:
                st.session_state.filter_state = json.loads(state_file.read_text(encoding="utf-8"))
            except Exception:
                st.session_state.filter_state = {}
        else:
            st.session_state.filter_state = {}
    return st.session_state.filter_state.get(key, default_val)

def set_static_filter_state(key: str, value: bool) -> None:
    """Writes persistent boolean state for UI filters directly."""
    if "filter_state" not in st.session_state:
        st.session_state.filter_state = {}
    st.session_state.filter_state[key] = value
    state_file = DATA_DIR / ".filter_state.json"
    try:
        state_file.write_text(json.dumps(st.session_state.filter_state, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print(f"[UI State Error] Could not save filter state: {e}", flush=True)

def handle_checkbox_change(key: str):
    """OnChange callback for Streamlit checkboxes."""
    val = st.session_state.get(key, True)
    set_static_filter_state(key, val)
    print(f"[UI Event] UI filter '{key}' changed to {val}", flush=True)

def handle_course_checkbox_change(c_key: str, files: list, c: str):
    """OnChange callback for full course checkbox."""
    val = st.session_state.get(c_key, True)
    set_static_filter_state(c_key, val)
    print(f"[UI Event] Course '{c}' bulk set to {val}", flush=True)
    # Align all document child checkboxes
    for f in files:
        f_key = f"chk_f_{c}_{f}"
        st.session_state[f_key] = val
        set_static_filter_state(f_key, val)

def handle_file_checkbox_change(f_key: str, c_key: str, files: list, c: str):
    """OnChange callback for a specific file checkbox."""
    val = st.session_state.get(f_key, True)
    set_static_filter_state(f_key, val)
    print(f"[UI Event] File '{f_key}' set to {val}", flush=True)
    
    # Auto-uncheck course if a file is unchecked
    if not val:
        st.session_state[c_key] = False
        set_static_filter_state(c_key, False)
    else:
        # Check if all files are now selected
        all_checked = all(st.session_state.get(f"chk_f_{c}_{f}", False) for f in files)
        if all_checked:
            st.session_state[c_key] = True
            set_static_filter_state(c_key, True)

# ---------------------- System Dialogs ----------------------


@st.dialog("⚙️ 偏好设置 / Settings", width="large")
def settings_dialog(agent: RAGAgent):
    settings = _get_settings()
    st.write("Configure your AI assistant.")
    
    with st.form("settings_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("LLM Provider")
            new_llm_api_key = st.text_input("LLM API Key", value=settings.llm_api_key or "", type="password")
            new_llm_base_url = st.text_input("LLM Base URL", value=settings.llm_base_url or "")
            new_llm_model = st.text_input("LLM Chat Model", value=settings.chat_model or "qwen-plus")
            new_vision_model = st.text_input("Vision Model", value=settings.vision_model or "qwen-vl-max")

        with col2:
            st.subheader("Embedding Provider")
            new_emb_provider = st.selectbox(
                "Embedding Provider",
                options=["gemini", "openai", "custom"],
                index=["gemini", "openai", "custom"].index(settings.embedding_provider) if settings.embedding_provider in ["gemini", "openai", "custom"] else 0
            )
            new_emb_api_key = st.text_input("Embedding API Key", value=settings.embedding_api_key or "", type="password")
            new_emb_base_url = st.text_input("Embedding Base URL", value=settings.embedding_base_url or "")
            new_emb_model = st.text_input("Embedding Model", value=settings.embedding_model or "")
        
        new_top_k = st.number_input("Top-K Retrievals", min_value=1, max_value=20, value=settings.top_k)

        if st.form_submit_button("Save defaults"):
            updated_dict = {
                "llm_api_key": new_llm_api_key,
                "llm_base_url": new_llm_base_url,
                "chat_model": new_llm_model,
                "vision_model": new_vision_model,
                "embedding_provider": new_emb_provider,
                "embedding_api_key": new_emb_api_key,
                "embedding_base_url": new_emb_base_url,
                "embedding_model": new_emb_model,
                "top_k": new_top_k,
                "history_limit": settings.history_limit,
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap
            }
            updated = normalize_settings(updated_dict)
            save_settings(updated)
            st.session_state.app_settings = asdict(updated)
            _apply_settings_to_agent(agent, updated)
            st.success("Settings saved!")
            st.rerun()


@st.dialog("➕ 新建课程与上传资料", width="large")
def new_course_dialog(agent: RAGAgent):
    tracker = FileTracker(str(DATA_DIR / "metadata.json"))
    st.markdown("创建一个新课程，并上传相关课件。")
    
    new_c_name = st.text_input("新课程名称", placeholder="例如：高级计算机网络")
    uploaded_files = st.file_uploader("支持文档，拖拽或点击上传", accept_multiple_files=True)
    
    if st.button("创建并上传索引...", type="primary", use_container_width=True) and uploaded_files and new_c_name:
        print(f"[UI Event] Creating new course: '{new_c_name}' with {len(uploaded_files)} files.", flush=True)
        course_dir = DATA_DIR / new_c_name
        course_dir.mkdir(parents=True, exist_ok=True)
        for uf in uploaded_files:
            with open(course_dir / uf.name, "wb") as f_out:
                f_out.write(uf.getbuffer())
                
        with st.spinner(f"正在为 {len(uploaded_files)} 个文件更新向量索引 (增量处理)..."):
            print(f"[UI Event] Executing internal index update.", flush=True)
            ok, detail = _rebuild_index_in_app(agent)
        
        if ok:
            print(f"[UI Event] Rebuild success. Appending new course default state.", flush=True)
            # Default newly created course to TRUE in selection
            set_static_filter_state(f"chk_c_{new_c_name}", True)
            st.success("全部文件上传并处理成功！")
            st.rerun()
        else:
            print(f"[UI Event] Rebuild failed: {detail}", flush=True)
            st.error(f"处理失败: {detail}")


@st.dialog("📚 课程资源管理", width="large")
def course_management_dialog(agent: RAGAgent):
    tracker = FileTracker(str(DATA_DIR / "metadata.json"))
    
    st.markdown("在这里管理当前的知识库体系。")
    
    manage_mode = st.toggle("🔒 管理模式 (开启后可删除)", value=False)
    
    current_courses = [d.name for d in DATA_DIR.iterdir() if d.is_dir() and d.name not in {"extracted_images", "__pycache__", ".DS_Store"}]
    if not current_courses:
        st.info("尚未创建任何课程。")
        
    for c in current_courses:
        c_dir = DATA_DIR / c
        files = [f for f in c_dir.iterdir() if f.is_file() and f.name != "metadata.json" and not f.name.startswith(".")]
        
        with st.expander(f"📁 {c} (包含 {len(files)} 份资料)"):
            if st.button("浏览 📂", key=f"browse_{c}", help="在资源管理器器中打开文件夹"):
                os.system(f"open '{c_dir}'")
            
            files_to_delete = []
            for f in files:
                if manage_mode:
                    # In manage mode, show a checkbox for multi-select delete
                    if st.checkbox(f"📄 `{f.name}`", key=f"sel_del_{c}_{f.name}"):
                        files_to_delete.append(f)
                else:
                    st.markdown(f"📄 `{f.name}`")
            
            if manage_mode and files_to_delete:
                if st.button(f"🗑 批量删除选中的 {len(files_to_delete)} 份资料", type="primary", key=f"bulk_del_{c}"):
                    try:
                        for f in files_to_delete:
                            if f.exists():
                                os.remove(f)
                            agent.vector_store.delete_by_filename(c, f.name)
                            tracker.remove_file(f.name, c)
                            st.session_state.pop(f"sel_del_{c}_{f.name}", None)
                            print(f"[UI Event] Deleted file {f.name} from {c}", flush=True)
                        st.success("批量移除成功！")
                        st.rerun()
                    except Exception as e:
                        st.error(f"删除失败: {e}")
                        print(f"[UI Event] Delete error: {e}", flush=True)
            
            st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)
            new_f = st.file_uploader(f"追加上传至 {c}", accept_multiple_files=True, key=f"up_{c}")
            if new_f and st.button("开始追加资料并索引", key=f"btn_up_{c}", type="secondary"):
                print(f"[UI Event] Uploading {len(new_f)} files into course '{c}'", flush=True)
                for uf in new_f:
                    with open(c_dir / uf.name, "wb") as f_out:
                        f_out.write(uf.getbuffer())
                with st.spinner(f"正在为 {c} 更新向量索引..."):
                    ok, detail = _rebuild_index_in_app(agent)
                if ok:
                    print(f"[UI Event] Add files to {c} index success", flush=True)
                    st.success("全部文件上传并处理成功！")
                    st.rerun()
                else:
                    print(f"[UI Event] Add files to {c} failed: {detail}", flush=True)
                    st.error(f"处理失败: {detail}")
                        
            if manage_mode:
                st.markdown("---")
                if st.button(f"🗑 删除课程体系 '{c}'", type="primary", key=f"del_c_{c}"):
                    for f in files:
                        agent.vector_store.delete_by_filename(c, f.name)
                        tracker.remove_file(f.name, c)
                    try:
                        shutil.rmtree(c_dir)
                        st.session_state["settings_toast"] = f"课程 {c} 已被永久删除。"
                    except Exception as e:
                        st.error(f"删除课程夹失败: {e}")
                    st.rerun()


@st.dialog("是否删除会话？")
def confirm_delete_dialog(session_path: Path):
    st.warning(f"确定要永久删除对话 '{session_path.stem}' 吗？")
    col1, col2 = st.columns(2)
    if col1.button("✅ 确认删除", type="primary", use_container_width=True):
        try:
            session_path.unlink()
            print(f"[UI Event] File deleted session {session_path.stem}", flush=True)
            if st.session_state.session_id == session_path.stem:
                st.session_state.messages = []
                st.session_state.retrieved_history = []
                st.session_state.session_title = "New Chat"
                st.session_state.session_id = None
            st.rerun()
        except Exception as e:
            st.error(f"删除失败: {e}")
    if col2.button("🚫 取消", use_container_width=True):
        st.rerun()

def sidebar(agent: RAGAgent, settings: AppSettings) -> Tuple[Optional[List[str]], Optional[List[str]]]:
    with st.sidebar:
        st.markdown("## 📚 CourseMate")
        st.caption("Your AI Course Assistant")
        st.divider()

        # Top Action Bar
        if st.button("⚙️ 偏好设置", use_container_width=True):
            settings_dialog(agent)
        if st.button("📚 课程资源管理", use_container_width=True):
            course_management_dialog(agent)
        if st.button("➕ 新建课程", use_container_width=True):
            new_course_dialog(agent)

        st.divider()

        # Cascading Context Filter (Tree Checkboxes)
        st.subheader("🎯 聚焦点 / Context")
        
        excluded = {"extracted_images", "__pycache__", ".DS_Store", "metadata.json"}
        courses = [d.name for d in DATA_DIR.iterdir() if d.is_dir() and d.name not in excluded]
        
        selected_courses = []
        selected_files = []
        
        # State tracking trick for expanding logic
        # Streamlit doesn't have native multi-level checkboxes that update perfectly unless wrapped nicely
        for c in courses:
            c_dir = DATA_DIR / c
            files = sorted([f.name for f in c_dir.iterdir() if f.is_file() and f.name not in excluded])
            
            with st.expander(f"📁 {c}"):
                c_key = f"chk_c_{c}"
                # Inject default value if Streamlit runtime doesn't have it initialized
                if c_key not in st.session_state:
                    st.session_state[c_key] = get_static_filter_state(c_key, True)
                
                c_checked = st.checkbox(
                    f"选中 {c} 全库", 
                    key=c_key, 
                    on_change=handle_course_checkbox_change, 
                    args=(c_key, files, c)
                )
                
                if c_checked:
                    selected_courses.append(c)
                
                # specific files selection
                st.markdown("<small style='color: grey;'>📌 在以下勾选的文件中检索：</small>", unsafe_allow_html=True)
                for f in files:
                    f_key = f"chk_f_{c}_{f}"
                    if f_key not in st.session_state:
                        # Auto-inherit from course if course is checked, otherwise default to its own state
                        default_f_state = True if c_checked else get_static_filter_state(f_key, True)
                        st.session_state[f_key] = default_f_state
                    
                    if st.checkbox(
                        f"📄 {f}", 
                        key=f_key, 
                        on_change=handle_file_checkbox_change, 
                        args=(f_key, c_key, files, c)
                    ):
                        if c not in selected_courses:
                            selected_courses.append(c)
                        selected_files.append(f)
                        
        # Deduplicate
        selected_courses = list(set(selected_courses))
        selected_files = list(set(selected_files))

        st.divider()

        # History Scrollable Area
        st.subheader("⏳ 会话与流")
        
        hist_container = st.container(height=400, border=False)
        session_files = sorted(SESSIONS_DIR.glob("*.json"), reverse=True)
        max_show = min(len(session_files), int(settings.history_limit))

        with hist_container:
            if st.button("✨ 开启新对话", type="primary" if st.session_state.session_id is None else "secondary", use_container_width=True):
                st.session_state.messages = []
                st.session_state.retrieved_history = []
                st.session_state.session_title = "New Chat"
                st.session_state.session_id = None
                st.rerun()
                
            for idx, f in enumerate(session_files[:max_show]):
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    title = data.get("title", f.stem)
                    msgs = data.get("messages", [])
                    q_count = len([m for m in msgs if m["role"] == "user"])
                    
                    is_active = (st.session_state.session_id == f.stem)
                    btn_type = "primary" if is_active else "secondary"
                    
                    # 缩短标题长度，防止换行让侧边栏撑大
                    safe_title = title[:9] + ".." if len(title) > 9 else title
                    display_text = f"{safe_title}({q_count}问)"
                    
                    hist_col, del_col = st.columns([5, 1.2])
                    if hist_col.button(display_text, key=f"hist_{f.stem}_{idx}", type=btn_type, use_container_width=True):
                        print(f"[UI Event] User loaded history session {f.stem}", flush=True)
                        load_session(f.stem)
                        st.rerun()
                        
                    with del_col:
                        if st.button("🗑️", key=f"del_btn_{f.stem}_{idx}", use_container_width=True, help="删除对话"):
                            confirm_delete_dialog(f)
                except Exception:
                    pass

    return selected_courses or None, selected_files or None


def _has_any_course_files() -> bool:
    excluded = {"extracted_images", "__pycache__", ".DS_Store", "metadata.json"}
    for d in DATA_DIR.iterdir():
        if not d.is_dir() or d.name in excluded:
            continue
        files = [f for f in d.iterdir() if f.is_file() and f.name not in excluded and not f.name.startswith(".")]
        if files:
            return True
    return False


def main() -> None:
    st.set_page_config(page_title="CourseMate", layout="wide", page_icon="✨")
    st.markdown("""
        <style>
        div[data-testid="stSidebar"] div[data-testid="stButton"] button p {
            justify-content: flex-start;
        }
        </style>
    """, unsafe_allow_html=True)
    _init_session_state()
    settings = _get_settings()
    with st.spinner("正在初始化 CourseMate，请稍候..."):
        agent = load_agent()
    _apply_settings_to_agent(agent, settings)

    toast = st.session_state.pop("settings_toast", "")
    if toast:
        st.toast(toast)

    # Sidebar handling
    course_filter, file_filter = sidebar(agent, settings)

    # Empty State Greeting
    if not st.session_state.messages:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.markdown(
            "<h1 style='text-align: center; color: #555; font-family: ui-sans-serif, system-ui;'>✨ 下午好，今天想学点什么？</h1>", 
            unsafe_allow_html=True
        )
        st.markdown(
            "<p style='text-align: center; color: #888; margin-top: 10px;'>左侧可以精确锁定教材索引范围。在下方发送您想探讨的学术问题，并可随时上传手写推导照片。</p>", 
            unsafe_allow_html=True
        )

    # Chat history display
    assistant_idx = 0
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message.get("image"):
                try:
                    st.image(base64.b64decode(message["image"]), width=300)
                except Exception:
                    pass
            st.markdown(message.get("content", ""))
            if message["role"] == "assistant" and assistant_idx < len(st.session_state.retrieved_history):
                docs = st.session_state.retrieved_history[assistant_idx]
                render.render_images(docs)
                render.render_sources(docs, idx)
                assistant_idx += 1

    # Vertical Pusher for compact styling
    st.markdown("<br>", unsafe_allow_html=True)
    
    placeholder = "关于这些知识点，你有什么疑惑？"
    chat_input_val = st.chat_input(placeholder, accept_file=True, file_type=["png", "jpg", "jpeg"])
    
    if chat_input_val:
        user_image_b64 = None
        prompt = ""
        
        if hasattr(chat_input_val, 'text') or hasattr(chat_input_val, 'files'):
            prompt = getattr(chat_input_val, 'text', "")
            files = getattr(chat_input_val, 'files', [])
            if files:
                user_image_b64 = base64.b64encode(files[0].getvalue()).decode("utf-8")
        elif isinstance(chat_input_val, dict):
            prompt = chat_input_val.get("text", "")
            files = chat_input_val.get("files", [])
            if files:
                user_image_b64 = base64.b64encode(files[0].getvalue()).decode("utf-8")
        else:
            prompt = str(chat_input_val)

        if not st.session_state.messages:
            st.session_state.session_title = prompt[:20] + ("..." if len(prompt) > 20 else "")
            
        with st.chat_message("user"):
            st.markdown(prompt)
            if user_image_b64:
                st.image(base64.b64decode(user_image_b64), width=350)
                
        msg = {"role": "user", "content": prompt}
        if user_image_b64:
            msg["image"] = user_image_b64
        st.session_state.messages.append(msg)
        
        print(f"[UI Event] Sending prompt: '{prompt}' to LLM", flush=True)

        with st.chat_message("assistant"):
            data_is_empty = not _has_any_course_files()
            no_course_selected = (not data_is_empty) and (not course_filter and not file_filter)

            if data_is_empty:
                answer = "您还尚未添加任何课程文件。请先在左侧上传课程资料后再提问。"
                docs = []
                st.markdown(answer)
            elif no_course_selected:
                answer = "您当前未选中任何课程。请先在左侧勾选课程或文件后再提问。"
                docs = []
                st.markdown(answer)
            else:
                history = st.session_state.messages[-7:-1]
                with st.spinner("AI 正在重组知识脉络..."):
                    print(f"[UI Event] Agent retrieving context with course_filter={course_filter} and file_filter={file_filter}", flush=True)
                    stream, docs = agent.answer_question_stream(
                        prompt, 
                        chat_history=history, 
                        course_names=course_filter, 
                        filenames=file_filter, 
                        user_image_b64=user_image_b64
                    )
                answer = st.write_stream(stream)
                if docs:
                    render.render_images(docs)
                    render.render_sources(docs, len(st.session_state.messages))
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.retrieved_history.append(docs)

        save_session()


if __name__ == "__main__":
    main()
