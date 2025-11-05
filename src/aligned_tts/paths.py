from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional


@lru_cache(maxsize=1)
def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def cosyvoice_root() -> Path:
    return project_root() / "CosyVoice"


def ensure_path_on_sys_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def default_prompt_wav() -> Path:
    return project_root() / "reference_speech" / "adult_female.wav"


def default_prompt_metadata() -> Path:
    return project_root() / "reference_speech" / "text.json"


def default_output_dir(subdir: Optional[str] = None) -> Path:
    base = project_root() / "generated_speech"
    if subdir:
        return base / subdir
    return base


__all__ = [
    "project_root",
    "cosyvoice_root",
    "ensure_path_on_sys_path",
    "default_prompt_wav",
    "default_prompt_metadata",
    "default_output_dir",
]
