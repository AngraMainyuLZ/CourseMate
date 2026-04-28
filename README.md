# CourseMate

[中文](README.zh-CN.md) | English

## 👋 Welcome to CourseMate

CourseMate is a course-focused RAG assistant built with Streamlit. It helps you turn lecture notes, slides, documents, source code, and configuration files into a searchable AI learning companion.

Ask questions, narrow retrieval to a course or file, upload a handwritten image, inspect cited sources, and generate practice questions from your own materials.

## ✨ Highlights

- 📚 **Multi-course knowledge base**: organize materials under `data/<CourseName>/` and search by course or individual file.
- 🔎 **Hybrid retrieval**: combines dense vector search with BM25 keyword retrieval for more robust results.
- 🧾 **Grounded answers**: answers are generated from retrieved course context and shown with file/page source references.
- 🖼️ **Multimodal support**: extracts images from PDF/PPTX materials and supports user-uploaded `png`, `jpg`, and `jpeg` images in chat.
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

4. **Add course materials**

   Put files under `data/<CourseName>/`, for example:

   ```text
   data/
     Probability/
       lecture1.pdf
       notes.md
     Programming/
       demo.py
       config.yaml
   ```

5. **Build or update the index**

   ```bash
   python scripts/process_data.py
   ```

6. **Run CourseMate**

   ```bash
   python scripts/run_streamlit.py
   ```

   Or run Streamlit directly:

   ```bash
   streamlit run UI/app.py
   ```

