from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

LOGGER = logging.getLogger("aligned_tts")


def setup_logging(level: int = logging.INFO) -> None:
    if LOGGER.handlers:
        return
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(name)s - %(message)s")
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
    LOGGER.setLevel(level)


def load_prompt_transcripts(metadata_path: Path) -> Dict[str, str]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Prompt metadata not found: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_prompt_text(prompt_name: str, metadata_path: Path) -> Optional[str]:
    prompts = load_prompt_transcripts(metadata_path)
    return prompts.get(prompt_name)


__all__ = ["LOGGER", "setup_logging", "load_prompt_transcripts", "resolve_prompt_text"]
