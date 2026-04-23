# CourseMate

CourseMate is a Streamlit-based RAG assistant for course learning materials.

It supports:
- PDF / PPTX / DOCX / TXT ingestion
- image extraction for PDF/PPTX pages
- hybrid retrieval (dense + BM25 fusion)
- source-grounded answers with file/page references
- optional image-generation flow for draw/diagram-style requests

## Key Features
- Multi-course filtering in sidebar (course/file level).
- Chat with optional uploaded image.
- Related source images + source snippets in answer panel.
- Runtime settings panel for LLM/Embedding providers and models.

## Project Structure
- `UI/`
  - `app.py`: Streamlit app entry
  - `render.py`: source/image rendering helpers
- `rag/`
  - `agent.py`: retrieval + prompt assembly + streaming answer
  - `image_gen.py`: draw-intent prompt optimization + image generation
  - `prompts.py`: system prompts and draw keywords
- `data_pipeline/`
  - `loader.py`: document/image loading
  - `splitter.py`: chunking strategy
  - `embeddings.py`: embedding client abstraction
  - `vector_store.py`: Chroma + BM25 hybrid retrieval
  - `tracker.py`: file hash tracking for incremental updates
- `scripts/`
  - `process_data.py`: build/rebuild vector index
  - `run_streamlit.py`: start Streamlit app
- `config.py`: paths/models/defaults
- `core/settings.py`: app settings persistence

## Quick Start
1. Create and activate virtual environment.
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Configure environment variables:
```bash
cp .env.example .env
```
Then fill API settings in `.env` (or use UI settings panel after launch).

4. (Optional but recommended) Build index:
```bash
python scripts/process_data.py
```

5. Run app:
```bash
python scripts/run_streamlit.py
```
or:
```bash
streamlit run UI/app.py
```

## Data Layout
Place course files under `data/<CourseName>/...`, for example:

```text
data/
  Probability/
    lec1.pdf
    lec2.pptx
  Signals/
    notes.docx
```

## Notes
- If data is empty, app can still start; it will ask user to upload course files before querying.
- If retrieval call fails (e.g., invalid API), error will be shown in chat stream.
- Sessions are saved under `sessions/`.
- Local data/settings/vector DB are ignored by git (see `.gitignore`) to reduce leakage risk.
