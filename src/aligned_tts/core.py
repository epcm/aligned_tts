from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path
from typing import Optional

import torchaudio

from .alignment import CTCForcedAligner, write_srt_file
from .paths import default_output_dir
from .tts_engine import CosyVoiceEngine
from .types import SegmentAlignment, TTSResult
from .utils import LOGGER, setup_logging
import time

setup_logging()


class AlignedTTS:
    """High-level facade combining CosyVoice synthesis and CTC-based forced alignment."""

    def __init__(
        self,
        *,
        cosyvoice_model_dir: Optional[str | Path] = None,
        fp16: bool = True,
        zero_shot_spk_id: str = "",
        alignment_language: str = "en",
        alignment_model_name: Optional[str] = None,
        alignment_device: Optional[str] = None,
        alignment_model_dir: Optional[str | Path] = None,
        output_dir: Optional[str | Path] = None,
    ) -> None:
        self.output_dir = Path(output_dir or default_output_dir())
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._tts = CosyVoiceEngine(
            model_dir=cosyvoice_model_dir,
            fp16=fp16,
            zero_shot_spk_id=zero_shot_spk_id,
        )
        self._aligner = CTCForcedAligner(
            language=alignment_language,
            model_name=alignment_model_name,
            device=alignment_device,
            model_dir=Path(alignment_model_dir) if alignment_model_dir else None,
        )

    def synthesize(
        self,
        text: str,
        *,
        prompt_wav: Optional[str | Path] = None,
        instruct_text: Optional[str] = None,
        manual_seed: Optional[int] = None,
        speed: float = 1.0,
        output_dir: Optional[str | Path] = None,
        audio_filename: Optional[str] = None,
        export_srt: bool = True,
    ) -> TTSResult:
        start_time = time.time()
        directory = Path(output_dir or self.output_dir)
        directory.mkdir(parents=True, exist_ok=True)

        base_name = audio_filename or _slugify(text)
        audio_path = directory / f"{base_name}.wav"

        waveform, sample_rate = self._tts.synthesize(
            text,
            prompt_wav=prompt_wav,
            instruct_text=instruct_text,
            manual_seed=manual_seed,
            speed=speed,
        )
        torchaudio.save(str(audio_path), waveform, sample_rate)
        LOGGER.info("Saved audio to %s", audio_path)

        alignment: list[SegmentAlignment] = list(
            self._aligner.align_audio(audio_path, text=text)
        )
        alignment_path = audio_path.with_suffix(".json")
        self._aligner.export_alignment_json(alignment, alignment_path)

        srt_path = None
        if export_srt:
            srt_path = audio_path.with_suffix(".srt")
            write_srt_file(alignment, srt_path)

        LOGGER.info("Alignment artifacts stored at %s", alignment_path)
        LOGGER.info("Total synthesis and alignment time: %.2f seconds, rtf: %.2f", time.time() - start_time, (time.time() - start_time) / (waveform.size(1) / sample_rate))
        return TTSResult(
            text=text,
            audio_path=audio_path,
            sample_rate=sample_rate,
            alignment=alignment,
            alignment_path=srt_path or alignment_path,
        )


def generate(
    text: str,
    *,
    output_dir: Optional[str | Path] = None,
    **kwargs,
) -> TTSResult:
    service = AlignedTTS(output_dir=output_dir, **kwargs)
    return service.synthesize(text, output_dir=output_dir)


def _slugify(text: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9]+", text)
    base = "-".join(tokens)[:48] or _dt.datetime.now().strftime("%Y%m%d")
    suffix = _dt.datetime.now().strftime("%H%M%S")
    return f"{base}-{suffix}"


__all__ = ["AlignedTTS", "generate"]
