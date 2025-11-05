from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Sequence

import torch

from . import whisperx_alignment as forced_alignment
from .types import CharAlignment, SegmentAlignment
from .utils import LOGGER, setup_logging

setup_logging()


class CTCForcedAligner:
    """Forced aligner built on wav2vec2-style CTC models without whisperx dependency."""

    def __init__(
        self,
        *,
        language: str = "en",
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        model_dir: Optional[Path] = None,
    ) -> None:
        self.language = language
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        LOGGER.info("Loading alignment model (lang=%s, device=%s)", language, self.device)
        self.model, self.metadata = forced_alignment.load_align_model(
            language_code=language,
            device=self.device,
            model_name=model_name,
            model_dir=model_dir,
        )
        self.sample_rate = forced_alignment.SAMPLE_RATE

    def norm_text(self, text: str) -> str:
        # remove <strong> tags and "*" and extra spaces
        return text.replace("<strong>", "").replace("</strong>", "").replace("*", "").strip()
    
    def align_audio(
        self,
        audio_path: Path,
        *,
        text: str,
    ) -> Sequence[SegmentAlignment]:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio not found: {audio_path}")
        if not text.strip():
            raise ValueError("Ground-truth text is required for alignment.")

        text = self.norm_text(text)
        waveform = forced_alignment.load_audio(str(audio_path), sample_rate=self.sample_rate)
        duration = waveform.shape[-1] / self.sample_rate
        transcript = [
            {"text": text, "start": 0.0, "end": float(duration)},
        ]

        LOGGER.info("Running forced alignment on %s seconds of audio", round(duration, 2))
        alignment_result = forced_alignment.align(
            transcript=transcript,
            model=self.model,
            align_model_metadata=self.metadata,
            audio=waveform,
            device=self.device,
            return_char_alignments=True,
        )

        return self._convert_segments(alignment_result.get("segments", []))

    def _convert_segments(self, segments: Sequence[dict]) -> Sequence[SegmentAlignment]:
        converted: List[SegmentAlignment] = []
        for segment in segments:
            text = segment.get("text", "")
            start = _float_or_none(segment.get("start"))
            end = _float_or_none(segment.get("end"))
            char_entries = []
            for char_info in segment.get("chars") or []:
                char = char_info.get("char", "")
                c_start = _float_or_none(char_info.get("start"))
                c_end = _float_or_none(char_info.get("end"))
                if c_start is None or c_end is None:
                    continue
                char_entries.append(
                    CharAlignment(
                        char=char,
                        start=c_start,
                        end=c_end,
                    )
                )
            converted.append(
                SegmentAlignment(
                    text=text,
                    start=start,
                    end=end,
                    chars=char_entries,
                )
            )
        return converted

    @staticmethod
    def export_alignment_json(alignment: Sequence[SegmentAlignment], output_path: Path) -> Path:
        payload = {
            "segments": [
                {
                    "text": segment.text,
                    "start": segment.start,
                    "end": segment.end,
                    "chars": [asdict(char) for char in segment.chars],
                }
                for segment in alignment
            ]
        }
        output_path = Path(output_path)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return output_path


def alignment_to_srt(alignment: Sequence[SegmentAlignment]) -> str:
    parts: List[str] = []
    index = 1
    for segment in alignment:
        base_text = segment.text or ""
        for i, char_alignment in enumerate(segment.chars):
            start_ts = _sec_to_srt_time(char_alignment.start)
            end_ts = _sec_to_srt_time(char_alignment.end)
            if i < len(base_text) and char_alignment.char.strip():
                highlighted = base_text[:i] + "<u>" + base_text[i] + "</u>" + base_text[i + 1 :]
            else:
                highlighted = base_text
            parts.append(f"{index}\n{start_ts} --> {end_ts}\n{highlighted}\n")
            index += 1
    return "\n".join(parts)


def write_srt_file(alignment: Sequence[SegmentAlignment], output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.write_text(alignment_to_srt(alignment), encoding="utf-8")
    return output_path


def _sec_to_srt_time(sec: float) -> str:
    total_ms = int(sec * 1000)
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


def _float_or_none(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return float(value)


__all__ = ["CTCForcedAligner", "alignment_to_srt", "write_srt_file"]
