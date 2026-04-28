"""Supported course material formats.

The loader has three parsing paths:
- paged/slided office formats with optional image extraction;
- Word documents;
- plain-text-like files, including source code and configuration files.
"""
from __future__ import annotations


PDF_FILE_EXTENSIONS = (".pdf",)
PRESENTATION_FILE_EXTENSIONS = (".pptx",)
WORD_FILE_EXTENSIONS = (".doc", ".docx")

TEXT_FILE_EXTENSIONS = (
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".tex",
    ".csv",
    ".tsv",
    ".log",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".properties",
    ".env",
    ".xml",
    ".html",
    ".htm",
    ".css",
    ".scss",
    ".sass",
    ".less",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".vue",
    ".svelte",
    ".py",
    ".ipynb",
    ".java",
    ".c",
    ".h",
    ".cpp",
    ".cc",
    ".cxx",
    ".hpp",
    ".hh",
    ".hxx",
    ".cs",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".kts",
    ".scala",
    ".sh",
    ".bash",
    ".zsh",
    ".ps1",
    ".bat",
    ".cmd",
    ".sql",
    ".r",
    ".m",
    ".jl",
    ".lua",
    ".pl",
    ".pm",
    ".dart",
    ".dockerfile",
)

SUPPORTED_FILE_EXTENSIONS = tuple(
    dict.fromkeys(
        PDF_FILE_EXTENSIONS
        + PRESENTATION_FILE_EXTENSIONS
        + WORD_FILE_EXTENSIONS
        + TEXT_FILE_EXTENSIONS
    )
)

