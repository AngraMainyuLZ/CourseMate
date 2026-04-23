"""Helper script to launch the CourseMate Streamlit UI."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    app_path = root / "UI" / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
