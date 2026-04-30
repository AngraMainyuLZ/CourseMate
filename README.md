# CourseMate

[中文](README.zh-CN.md) | English

## 👋 Welcome to CourseMate

CourseMate is a course-focused RAG assistant built with Streamlit. It helps you turn lecture notes, slides, documents, source code, and configuration files into a searchable AI learning companion.

Ask questions, narrow retrieval to a course or file, upload a handwritten image, inspect cited sources, and generate practice questions from your own materials.

## ✨ Highlights

- 📚 **Multi-course knowledge base**: organize materials under `data/<CourseName>/` and search by course or individual file.
- 🔎 **Hybrid retrieval**: combines dense vector search with BM25 keyword retrieval for more robust results.
- 🧾 **Grounded answers**: answers are generated from retrieved course context and shown with file/page source references.
- 🖼️ **Multimodal & Visual RAG**: extracts images from PDF/PPTX materials and supports user-uploaded `png`, `jpg`, and `jpeg` images. Furthermore, an advanced **Visual RAG (PDF only)** mode lets models directly 'read' the original rendered PDF pages to flawlessly comprehend complex mathematical formulas and charts. *(Note: For PPTX slides, please export them to PDF format first before uploading to leverage this advanced feature.)*
- 🧠 **Auto quiz mode**: generates structured practice questions, answers, difficulty labels, and citations from selected materials.
- 🎨 **Diagram/image generation flow**: drawing-style requests can be converted into grounded image prompts and sent to an image model.
- ⚙️ **Runtime settings**: configure LLM, vision model, image model, embedding provider, Top-K retrieval, and quiz size from the app.
- 🧩 **Broad file ingestion**: supports PDF, PPTX, DOC/DOCX, text files, Markdown, JSON/YAML/TOML, web files, scripts, and common source-code formats.

## 🚀 Quick Start

1. **Create and activate a Python environment**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API settings**

   ```bash
   copy .env.example .env
   ```

   Fill in your model and embedding API settings in `.env`, or configure them later in the app settings panel.

4. **Run CourseMate**

   ```bash
   python scripts/run_streamlit.py
   ```

   Or run Streamlit directly:

   ```bash
   streamlit run UI/app.py
   ```

5. **Manage courses & update index in UI**

   With the Streamlit app launched, use the sidebar to create new courses, upload existing files (or manually put files under `data/<CourseName>/`), and seamlessly trigger index updates inside the interface—no CLI scripts needed!

