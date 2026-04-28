# CourseMate

中文 | [English](README.md)

## 👋 欢迎使用 CourseMate

CourseMate 是一个面向课程学习资料的 RAG 课程助教，基于 Streamlit 构建。它可以把课件、讲义、文档、代码源文件和配置文件转化为可检索的 AI 学习助手。

你可以围绕课程资料提问，限定检索到某门课或某个文件，上传手写推导图片，查看回答引用来源，并从自己的材料中自动生成练习题。

## ✨ 亮点功能

- 📚 **多课程知识库**：按 `data/<课程名>/` 组织资料，并支持按课程或单个文件限定检索范围。
- 🔎 **混合检索**：结合向量检索和 BM25 关键词检索，提高召回稳定性。
- 🧾 **有依据的回答**：回答基于检索到的课程上下文生成，并展示文件和页码来源。
- 🖼️ **多模态支持**：可从 PDF/PPTX 中抽取图片，也支持在聊天中上传 `png`、`jpg`、`jpeg` 图片。
- 🧠 **自动出题模式**：基于选定资料生成题目、答案、难度标签和引用来源。
- 🎨 **图示/图片生成流程**：识别画图类请求后，可结合课程上下文生成图片模型提示词。
- ⚙️ **运行时设置**：可在应用中配置 LLM、视觉模型、图片模型、Embedding Provider、Top-K 和出题数量。
- 🧩 **广泛的文件解析支持**：支持 PDF、PPTX、DOC/DOCX、文本、Markdown、JSON/YAML/TOML、网页文件、脚本和常见代码源文件格式。

## 🚀 Quick Start

1. **创建并激活 Python 环境**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. **安装依赖**

   ```bash
   pip install -r requirements.txt
   ```

3. **配置 API**

   ```bash
   copy .env.example .env
   ```

   在 `.env` 中填写模型和 embedding API 配置；也可以启动应用后在设置面板中配置。

4. **添加课程资料**

   将文件放到 `data/<课程名>/` 下，例如：

   ```text
   data/
     随机过程/
       lec1.pdf
       notes.md
     程序设计/
       demo.py
       config.yaml
   ```

5. **构建或更新索引**

   ```bash
   python scripts/process_data.py
   ```

6. **启动 CourseMate**

   ```bash
   python scripts/run_streamlit.py
   ```

   或者直接运行 Streamlit：

   ```bash
   streamlit run UI/app.py
   ```

