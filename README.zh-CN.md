# CourseMate

中文 | [English](README.md)

## 👋 欢迎使用 CourseMate

CourseMate 是一个面向课程学习资料的 RAG 课程助教，基于 Streamlit 构建。它可以把课件、讲义、文档、代码源文件和配置文件转化为可检索的 AI 学习助手。

你可以围绕课程资料提问，限定检索到某门课或某个文件，上传手写推导图片，查看回答引用来源，并从自己的材料中自动生成练习题。

## ✨ 亮点功能

- 📚 **多课程知识库**：按 `data/<课程名>/` 组织资料，并支持按课程或单个文件限定检索范围。
- 🔎 **混合检索**：结合向量检索和 BM25 关键词检索，提高召回稳定性。
- 🧾 **有依据的回答**：回答基于检索到的课程上下文生成，并展示文件和页码来源。
- 🖼️ **多模态与原图增强阅读 (Visual RAG)**：不仅可从 PDF/PPTX 中抽取小碎片图片和在聊天中上传图像，还支持前沿的 **PDF 原图增强阅读 (仅限 PDF)**。它允许大模型直接“看”高清文档原汁原味的版面，从而完美保留和理解繁杂的数学公式与复杂图表。*(注意：由于渲染难度限制，对于 PPTX 课件，若想激活此原图阅读特性，请在上传系统前自行将其另存为 PDF 格式！)*
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

4. **启动 CourseMate**

   ```bash
   python scripts/run_streamlit.py
   ```

   或者直接运行 Streamlit：

   ```bash
   streamlit run UI/app.py
   ```

5. **在 UI 面板中管理课程与索引**

   启动后，直接在左侧边栏操作：你可以一键新建课程、上传课程文件（或将文件丢入 `data/<课程名>/` 内），然后在界面中点击按钮直接完成文档向量索引构建及更新 —— 无需再频繁使用命令行执行更新脚本啦！

