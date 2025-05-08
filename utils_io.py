#utils_io.py
"""Generic file I/O helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import pandas as pd

__all__: Final = ["load_dataframe", "save_dataframe"]


def load_dataframe(file_path: str | Path) -> pd.DataFrame:
    path = Path(file_path)
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".json":
        return pd.DataFrame(json.loads(path.read_text()))
    if ext == ".ndjson":
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported file type: {ext}")


def save_dataframe(df: pd.DataFrame, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
