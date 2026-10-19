import os
from pathlib import Path

SUBDIRS = ["drop", "processed", "rejected", "db", "cache", "models"]

def data_root() -> Path:
    p = os.environ.get("URECA_DATA_ROOT")
    if not p:
        raise RuntimeError("Set URECA_DATA_ROOT to the URECA_project_root folder.")
    root = Path(p)
    if not root.is_dir():
        raise FileNotFoundError(f"Data root not found: {root}")
    return root

def ensure_dirs() -> dict[str, Path]:
    root = data_root()
    paths = {name: root / name for name in SUBDIRS}
    for p in paths.values():
        p.mkdir(exist_ok=True)
    return paths