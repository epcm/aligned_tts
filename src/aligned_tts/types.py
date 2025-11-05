from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence


@dataclass(slots=True)
class CharAlignment:
    """Character-level alignment entry."""

    char: str
    start: float
    end: float


@dataclass(slots=True)
class SegmentAlignment:
    """Segment-level alignment data."""

    text: str
    start: Optional[float]
    end: Optional[float]
    chars: Sequence[CharAlignment] = field(default_factory=list)


@dataclass(slots=True)
class TTSResult:
    """Container for synthesized audio and alignment."""

    text: str
    audio_path: Path
    sample_rate: int
    alignment: Sequence[SegmentAlignment]
    alignment_path: Optional[Path] = None

    def iter_char_alignment(self) -> List[CharAlignment]:
        chars: List[CharAlignment] = []
        for segment in self.alignment:
            chars.extend(segment.chars)
        return chars


__all__ = ["CharAlignment", "SegmentAlignment", "TTSResult"]
