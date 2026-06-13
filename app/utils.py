from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def file_status(path: Path) -> Dict[str, Any]:
    """Report existence / mtime / size for a generated artifact on disk."""
    if not path.exists():
        return {"exists": False, "updated_at": None, "size_kb": None}
    stat = path.stat()
    return {
        "exists": True,
        "updated_at": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "size_kb": round(stat.st_size / 1024, 1),
    }
