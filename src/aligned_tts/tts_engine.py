from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch

from .paths import cosyvoice_root, ensure_path_on_sys_path, default_prompt_metadata, default_prompt_wav
from .utils import LOGGER, resolve_prompt_text, setup_logging

setup_logging()

COSYVOICE_DIR = cosyvoice_root()
MATCHA_DIR = COSYVOICE_DIR / "third_party" / "Matcha-TTS"

if COSYVOICE_DIR.exists():
    ensure_path_on_sys_path(COSYVOICE_DIR)
    if MATCHA_DIR.exists():
        ensure_path_on_sys_path(MATCHA_DIR)
else:
    LOGGER.warning("CosyVoice directory not found at %s", COSYVOICE_DIR)

try:
    from cosyvoice.cli.cosyvoice import CosyVoice2
    from cosyvoice.utils.common import set_all_random_seed
    from cosyvoice.utils.file_utils import load_wav
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError(
        "Failed to import CosyVoice dependencies. Ensure CosyVoice is available under the project root."
    ) from exc


class CosyVoiceEngine:
    """Wrapper around CosyVoice2 for instruct-style synthesis."""

    def __init__(
        self,
        model_dir: Optional[Path | str] = None,
        fp16: bool = True,
        zero_shot_spk_id: str = "",
    ) -> None:
        self.model_dir = Path(model_dir or (COSYVOICE_DIR / "pretrained_models" / "CosyVoice2-0.5B"))
        self.fp16 = fp16
        self.zero_shot_spk_id = zero_shot_spk_id
        self._prompt_cache: Dict[str, torch.Tensor] = {}

        if not self.model_dir.exists():
            raise FileNotFoundError(f"CosyVoice model directory not found: {self.model_dir}")

        LOGGER.info("Loading CosyVoice2 weights from %s", self.model_dir)
        self._engine = CosyVoice2(
            str(self.model_dir),
            load_jit=False,
            load_trt=False,
            load_vllm=False,
            fp16=fp16,
        )

        self.sample_rate = self._engine.sample_rate

    def synthesize(
        self,
        text: str,
        *,
        prompt_wav: Optional[Path | str] = None,
        instruct_text: Optional[str] = None,
        speed: float = 1.0,
        manual_seed: int = 0,
        stream: bool = False,
        zero_shot_spk_id: Optional[str] = None,
    ) -> tuple[torch.Tensor, int]:
        if not text.strip():
            raise ValueError("Input text must be non-empty")

        prompt_path = Path(prompt_wav)
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt audio not found: {prompt_path}")

        prompt_audio = load_wav(str(prompt_path), 16000)

        instruct = instruct_text or ""

        outputs: list[torch.Tensor] = []
        set_all_random_seed(manual_seed)
        LOGGER.info(f"text: {text}, instruct: {instruct}, manual_seed: {manual_seed}, stream: {stream}, speed: {speed}")
        for chunk in self._engine.inference_instruct2(
            text,
            instruct,
            prompt_audio,
            zero_shot_spk_id=zero_shot_spk_id if zero_shot_spk_id is not None else self.zero_shot_spk_id,
            stream=stream,
            speed=speed,
        ):
            outputs.append(chunk["tts_speech"].detach().cpu())

        if not outputs:
            raise RuntimeError("CosyVoice returned no audio output")

        waveform = torch.cat(outputs, dim=1)
        return waveform, self.sample_rate


__all__ = ["CosyVoiceEngine"]
