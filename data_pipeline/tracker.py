import json
import hashlib
from pathlib import Path
from typing import Dict, Any

class FileTracker:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.registry: Dict[str, Dict[str, Any]] = self._load()

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if self.db_path.exists():
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self.registry, f, ensure_ascii=False, indent=2)

    def get_file_hash(self, file_path: str) -> str:
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def has_changed(self, file_path: str, course_name: str) -> bool:
        path_str = str(file_path)
        if not Path(path_str).exists():
            return True
        current_hash = self.get_file_hash(path_str)
        key = f"{course_name}:{Path(path_str).name}"
        if key in self.registry:
            if self.registry[key].get("hash") == current_hash:
                return False
        return True

    def mark_processed(self, file_path: str, course_name: str) -> None:
        path_str = str(file_path)
        current_hash = self.get_file_hash(path_str)
        key = f"{course_name}:{Path(path_str).name}"
        self.registry[key] = {
            "hash": current_hash,
            "filename": Path(path_str).name,
            "course_name": course_name
        }
        self._save()

    def remove_file(self, filename: str, course_name: str) -> None:
        key = f"{course_name}:{filename}"
        if key in self.registry:
            del self.registry[key]
            self._save()
