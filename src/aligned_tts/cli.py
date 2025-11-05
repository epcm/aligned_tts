from __future__ import annotations

import argparse
from pathlib import Path

from .core import AlignedTTS

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CosyVoice2 TTS with char-level alignment")
    parser.add_argument("text", help="Text to synthesise")
    parser.add_argument("--output-dir", type=Path, default=None, help="Destination directory for artefacts")
    parser.add_argument("--audio-filename", default=None, help="Optional base filename (without extension)")
    parser.add_argument("--no-srt", action="store_true", help="Skip SRT export")
    return parser

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    service = AlignedTTS()
    result = service.synthesize(
        args.text,
        output_dir=args.output_dir,
        audio_filename=args.audio_filename,
        export_srt=not args.no_srt,
    )

    print(f"Audio: {result.audio_path}")
    print(f"Alignment: {result.alignment_path}")

if __name__ == "__main__":
    main()
