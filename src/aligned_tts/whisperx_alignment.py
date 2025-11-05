from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch import Tensor
from torchaudio import functional as F
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

import nltk
from nltk.data import load as nltk_load

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
LANGUAGES_WITHOUT_SPACES = ["ja", "zh"]

DEFAULT_ALIGN_MODELS_TORCH: Dict[str, str] = {
    "en": "WAV2VEC2_ASR_BASE_960H",
    "fr": "VOXPOPULI_ASR_BASE_10K_FR",
    "de": "VOXPOPULI_ASR_BASE_10K_DE",
    "es": "VOXPOPULI_ASR_BASE_10K_ES",
    "it": "VOXPOPULI_ASR_BASE_10K_IT",
}

DEFAULT_ALIGN_MODELS_HF: Dict[str, str] = {
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
    "uk": "Yehor/wav2vec2-xls-r-300m-uk-with-small-lm",
    "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
    "ar": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
    "cs": "comodoro/wav2vec2-xls-r-300m-cs-250",
    "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    "pl": "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
    "hu": "jonatasgrosman/wav2vec2-large-xlsr-53-hungarian",
    "fi": "jonatasgrosman/wav2vec2-large-xlsr-53-finnish",
    "fa": "jonatasgrosman/wav2vec2-large-xlsr-53-persian",
    "el": "jonatasgrosman/wav2vec2-large-xlsr-53-greek",
    "tr": "mpoyraz/wav2vec2-xls-r-300m-cv7-turkish",
    "da": "saattrupdan/wav2vec2-xls-r-300m-ftspeech",
    "he": "imvladikon/wav2vec2-xls-r-300m-hebrew",
    "vi": "nguyenvulebinh/wav2vec2-base-vi-vlsp2020",
    "ko": "kresnik/wav2vec2-large-xlsr-korean",
    "ur": "kingabzpro/wav2vec2-large-xls-r-300m-Urdu",
    "te": "anuragshas/wav2vec2-large-xlsr-53-telugu",
    "hi": "theainerd/Wav2Vec2-large-xlsr-hindi",
    "ca": "softcatala/wav2vec2-large-xlsr-catala",
    "ml": "gvs/wav2vec2-large-xlsr-malayalam",
    "no": "NbAiLab/nb-wav2vec2-1b-bokmaal-v2",
    "nn": "NbAiLab/nb-wav2vec2-1b-nynorsk",
    "sk": "comodoro/wav2vec2-xls-r-300m-sk-cv8",
    "sl": "anton-l/wav2vec2-large-xlsr-53-slovenian",
    "hr": "classla/wav2vec2-xls-r-parlaspeech-hr",
    "ro": "gigant/romanian-wav2vec2",
    "eu": "stefan-it/wav2vec2-large-xlsr-53-basque",
    "gl": "ifrz/wav2vec2-large-xlsr-galician",
    "ka": "xsway/wav2vec2-large-xlsr-georgian",
    "lv": "jimregan/wav2vec2-large-xlsr-latvian-cv",
    "tl": "Khalsuu/filipino-wav2vec2-l-xls-r-300m-official",
    "sv": "KBLab/wav2vec2-large-voxrex-swedish",
}

PUNKT_LANGUAGES: Dict[str, str] = {
    "en": "english",
    "fr": "french",
    "de": "german",
    "es": "spanish",
    "it": "italian",
    "pt": "portuguese",
    "nl": "dutch",
    "ca": "catalan",
    "cs": "czech",
    "da": "danish",
    "fi": "finnish",
    "hu": "hungarian",
    "no": "norwegian",
    "pl": "polish",
    "sk": "slovak",
    "sl": "slovene",
    "sv": "swedish",
}

SingleSegment = Dict[str, Union[str, float]]
SingleAlignedSegment = Dict[str, Any]
SingleWordSegment = Dict[str, Any]
SegmentData = Dict[str, Any]
AlignedTranscriptionResult = Dict[str, List[Dict[str, Any]]]


def load_audio(
    audio: Union[str, Path, np.ndarray, Tensor],
    sample_rate: int = SAMPLE_RATE,
) -> Tensor:
    if isinstance(audio, (str, Path)):
        waveform, sr = torchaudio.load(str(audio))
    elif isinstance(audio, np.ndarray):
        waveform = torch.from_numpy(audio)
        sr = sample_rate
    elif torch.is_tensor(audio):
        waveform = audio
        sr = sample_rate
    else:
        raise TypeError(f"Unsupported audio type: {type(audio)}")

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != sample_rate:
        waveform = F.resample(waveform, sr, sample_rate)

    return waveform


def interpolate_nans(series: pd.Series, method: str = "nearest") -> pd.Series:
    series = series.astype(float)
    interpolated = series.interpolate(method=method, limit_direction="both")
    return interpolated


def load_align_model(
    language_code: str,
    device: str,
    model_name: Optional[str] = None,
    model_dir: Optional[Union[str, Path]] = None,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    if model_name is None:
        if language_code in DEFAULT_ALIGN_MODELS_TORCH:
            model_name = DEFAULT_ALIGN_MODELS_TORCH[language_code]
        elif language_code in DEFAULT_ALIGN_MODELS_HF:
            model_name = DEFAULT_ALIGN_MODELS_HF[language_code]
        else:
            raise ValueError(
                f"No default alignment model for language '{language_code}'. "
                "Provide a Hugging Face model name or torchaudio pipeline."
            )

    if model_name in torchaudio.pipelines.__all__:
        bundle = torchaudio.pipelines.__dict__[model_name]
        align_model = bundle.get_model(dl_kwargs={"model_dir": model_dir}).to(device)
        labels = bundle.get_labels()
        align_dictionary = {c.lower(): i for i, c in enumerate(labels)}
        metadata = {"language": language_code, "dictionary": align_dictionary, "type": "torchaudio"}
    else:
        processor = Wav2Vec2Processor.from_pretrained(model_name, cache_dir=model_dir)
        align_model = Wav2Vec2ForCTC.from_pretrained(model_name, cache_dir=model_dir).to(device)
        labels = processor.tokenizer.get_vocab()
        align_dictionary = {char.lower(): idx for char, idx in labels.items()}
        metadata = {
            "language": language_code,
            "dictionary": align_dictionary,
            "type": "huggingface",
            "processor": processor,
        }

    align_model.eval()
    return align_model, metadata


def align(
    transcript: Iterable[SingleSegment],
    model: torch.nn.Module,
    align_model_metadata: Dict[str, Any],
    audio: Union[str, np.ndarray, Tensor, Path],
    device: str,
    interpolate_method: str = "nearest",
    return_char_alignments: bool = False,
) -> AlignedTranscriptionResult:
    waveform = load_audio(audio)
    if len(waveform.shape) == 1:
        waveform = waveform.unsqueeze(0)

    max_duration = waveform.shape[1] / SAMPLE_RATE
    model_dictionary = align_model_metadata["dictionary"]
    model_lang = align_model_metadata["language"]
    model_type = align_model_metadata["type"]

    transcript = list(transcript)
    total_segments = len(transcript)

    segment_data: Dict[int, SegmentData] = {}
    for sdx, segment in enumerate(transcript):
        text = str(segment["text"])
        num_leading = len(text) - len(text.lstrip())
        num_trailing = len(text) - len(text.rstrip())

        if model_lang in LANGUAGES_WITHOUT_SPACES:
            per_word = text
        else:
            per_word = text.split(" ")

        clean_char: List[str] = []
        clean_cdx: List[int] = []
        for cdx, char in enumerate(text):
            char_ = char.lower()
            if model_lang not in LANGUAGES_WITHOUT_SPACES:
                char_ = char_.replace(" ", "|")

            if cdx < num_leading or cdx > len(text) - num_trailing - 1:
                continue
            if char_ in model_dictionary:
                clean_char.append(char_)
                clean_cdx.append(cdx)
            else:
                clean_char.append("*")
                clean_cdx.append(cdx)

        clean_wdx: List[int] = []
        for wdx, word in enumerate(per_word):
            word_chars = word.lower()
            if any(c in model_dictionary for c in word_chars):
                clean_wdx.append(wdx)
            else:
                clean_wdx.append(wdx)

        punkt_lang = PUNKT_LANGUAGES.get(model_lang, "english")
        try:
            sentence_splitter = nltk_load(f"tokenizers/punkt/{punkt_lang}.pickle")
        except LookupError:
            nltk.download("punkt", quiet=True)
            sentence_splitter = nltk_load(f"tokenizers/punkt/{punkt_lang}.pickle")
        sentence_spans = list(sentence_splitter.span_tokenize(text))

        segment_data[sdx] = {
            "clean_char": clean_char,
            "clean_cdx": clean_cdx,
            "clean_wdx": clean_wdx,
            "sentence_spans": sentence_spans,
        }

    aligned_segments: List[SingleAlignedSegment] = []

    for sdx, segment in enumerate(transcript):
        t1 = float(segment["start"])
        t2 = float(segment["end"])
        text = str(segment["text"])

        aligned_seg: SingleAlignedSegment = {
            "start": t1,
            "end": t2,
            "text": text,
            "words": [],
            "chars": [] if return_char_alignments else None,
        }

        if not segment_data[sdx]["clean_char"]:
            logger.warning(
                "No characters from transcript found in dictionary; skipping alignment for segment '%s'",
                text,
            )
            aligned_segments.append(aligned_seg)
            continue

        if t1 >= max_duration:
            logger.warning(
                "Segment start (%.2f) beyond audio duration (%.2f); skipping segment '%s'",
                t1,
                max_duration,
                text,
            )
            aligned_segments.append(aligned_seg)
            continue

        text_clean = "".join(segment_data[sdx]["clean_char"])
        tokens = [model_dictionary.get(c, -1) for c in text_clean]

        f1 = max(int(t1 * SAMPLE_RATE), 0)
        f2 = min(int(t2 * SAMPLE_RATE), waveform.shape[1])
        if f2 <= f1:
            f2 = waveform.shape[1]

        waveform_segment = waveform[:, f1:f2]
        if waveform_segment.shape[-1] < 400:
            lengths = torch.as_tensor([waveform_segment.shape[-1]], device=device)
            waveform_segment = torch.nn.functional.pad(
                waveform_segment, (0, 400 - waveform_segment.shape[-1])
            )
        else:
            lengths = None

        with torch.inference_mode():
            if model_type == "torchaudio":
                emissions, _ = model(waveform_segment.to(device), lengths=lengths)
            elif model_type == "huggingface":
                emissions = model(waveform_segment.to(device)).logits
            else:
                raise NotImplementedError(f"Unsupported alignment model type '{model_type}'")
            emissions = torch.log_softmax(emissions, dim=-1)

        emission = emissions[0].cpu().detach()

        blank_id = 0
        for char, code in model_dictionary.items():
            if char in {"[pad]", "<pad>"}:
                blank_id = code

        trellis = get_trellis(emission, tokens, blank_id)
        path = backtrack_beam(trellis, emission, tokens, blank_id, beam_width=2)

        if path is None:
            logger.warning("Failed to backtrack for segment '%s'; skipping", text)
            aligned_segments.append(aligned_seg)
            continue

        char_segments = merge_repeats(path, text_clean)

        duration = t2 - t1 if t2 > t1 else waveform_segment.size(-1) / SAMPLE_RATE
        ratio = duration * waveform_segment.size(0) / max(trellis.size(0) - 1, 1)

        char_segments_arr = []
        word_idx = 0
        for cdx, char in enumerate(text):
            start = end = score = None
            if cdx in segment_data[sdx]["clean_cdx"]:
                idx = segment_data[sdx]["clean_cdx"].index(cdx)
                char_seg = char_segments[idx]
                start = round(char_seg.start * ratio + t1, 3)
                end = round(char_seg.end * ratio + t1, 3)
                score = round(char_seg.score, 3)

            char_segments_arr.append(
                {"char": char, "start": start, "end": end, "score": score, "word-idx": word_idx}
            )

            if model_lang in LANGUAGES_WITHOUT_SPACES:
                word_idx += 1
            elif cdx == len(text) - 1 or text[cdx + 1] == " ":
                word_idx += 1

        char_segments_df = pd.DataFrame(char_segments_arr)

        aligned_subsegments = []
        char_segments_df["sentence-idx"] = None
        for sdx2, (sstart, send) in enumerate(segment_data[sdx]["sentence_spans"]):
            curr_chars = char_segments_df.loc[
                (char_segments_df.index >= sstart) & (char_segments_df.index <= send)
            ].copy()
            char_segments_df.loc[
                (char_segments_df.index >= sstart) & (char_segments_df.index <= send), "sentence-idx"
            ] = sdx2

            sentence_text = text[sstart:send]
            sentence_start = curr_chars["start"].min()
            non_space_chars = curr_chars[curr_chars["char"] != " "]
            sentence_end = non_space_chars["end"].max()
            sentence_words = []

            for word_index in curr_chars["word-idx"].unique():
                word_chars = curr_chars.loc[curr_chars["word-idx"] == word_index]
                word_text = "".join(word_chars["char"].tolist()).strip()
                if not word_text:
                    continue

                word_chars = word_chars[word_chars["char"] != " "]
                word_start = word_chars["start"].min()
                word_end = word_chars["end"].max()
                word_score = round(word_chars["score"].mean(), 3)

                word_segment: Dict[str, Any] = {"word": word_text}
                if not math.isnan(word_start):
                    word_segment["start"] = word_start
                if not math.isnan(word_end):
                    word_segment["end"] = word_end
                if not math.isnan(word_score):
                    word_segment["score"] = word_score
                sentence_words.append(word_segment)

            aligned_subsegments.append(
                {
                    "text": sentence_text,
                    "start": sentence_start,
                    "end": sentence_end,
                    "words": sentence_words,
                }
            )

            if return_char_alignments:
                char_records = curr_chars[["char", "start", "end", "score"]].copy()
                char_records.fillna(-1, inplace=True)
                char_payload = []
                for record in char_records.to_dict("records"):
                    payload = {k: v for k, v in record.items() if v != -1}
                    char_payload.append(payload)
                aligned_subsegments[-1]["chars"] = char_payload

        aligned_df = pd.DataFrame(aligned_subsegments)
        if not aligned_df.empty:
            aligned_df["start"] = interpolate_nans(aligned_df["start"], method=interpolate_method)
            aligned_df["end"] = interpolate_nans(aligned_df["end"], method=interpolate_method)

            agg_dict: Dict[str, Any] = {"text": " ".join, "words": "sum"}
            if model_lang in LANGUAGES_WITHOUT_SPACES:
                agg_dict["text"] = "".join
            if return_char_alignments:
                agg_dict["chars"] = "sum"

            aligned_df = aligned_df.groupby(["start", "end"], as_index=False).agg(agg_dict)
            aligned_segments.extend(aligned_df.to_dict("records"))
        else:
            aligned_segments.append(aligned_seg)

    word_segments: List[SingleWordSegment] = []
    for segment in aligned_segments:
        word_segments.extend(segment.get("words", []))

    return {"segments": aligned_segments, "word_segments": word_segments}


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


@dataclass
class Path:
    points: List[Point]
    score: float


@dataclass
class BeamState:
    token_index: int
    time_index: int
    score: float
    path: List[Point]


def get_trellis(emission: Tensor, tokens: List[int], blank_id: int = 0) -> Tensor:
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    trellis = torch.zeros((num_frame, num_tokens))
    trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
    trellis[0, 1:] = -float("inf")
    trellis[-num_tokens + 1 :, 0] = float("inf")

    for t in range(num_frame - 1):
        trellis[t + 1, 1:] = torch.maximum(
            trellis[t, 1:] + emission[t, blank_id],
            trellis[t, :-1] + get_wildcard_emission(emission[t], tokens[1:], blank_id),
        )
    return trellis


def get_wildcard_emission(frame_emission: Tensor, tokens: List[int], blank_id: int) -> Tensor:
    tokens_tensor = torch.tensor(tokens)
    wildcard_mask = tokens_tensor == -1
    regular_scores = frame_emission[tokens_tensor.clamp(min=0).long()]

    max_valid = frame_emission.clone()
    max_valid[blank_id] = float("-inf")
    max_valid_score = max_valid.max()

    return torch.where(wildcard_mask, max_valid_score, regular_scores)


def backtrack_beam(
    trellis: Tensor,
    emission: Tensor,
    tokens: List[int],
    blank_id: int = 0,
    beam_width: int = 5,
) -> Optional[List[Point]]:
    T, J = trellis.size(0) - 1, trellis.size(1) - 1

    init_state = BeamState(
        token_index=J,
        time_index=T,
        score=trellis[T, J],
        path=[Point(J, T, emission[T, blank_id].exp().item())],
    )
    beams = [init_state]

    while beams and beams[0].token_index > 0:
        next_beams: List[BeamState] = []
        for beam in beams:
            t, j = beam.time_index, beam.token_index
            if t <= 0:
                continue

            p_stay = emission[t - 1, blank_id]
            p_change = get_wildcard_emission(emission[t - 1], [tokens[j]], blank_id)[0]

            stay_score = trellis[t - 1, j]
            change_score = trellis[t - 1, j - 1] if j > 0 else float("-inf")

            if not math.isinf(stay_score):
                new_path = beam.path.copy()
                new_path.append(Point(j, t - 1, p_stay.exp().item()))
                next_beams.append(
                    BeamState(token_index=j, time_index=t - 1, score=stay_score, path=new_path)
                )

            if j > 0 and not math.isinf(change_score):
                new_path = beam.path.copy()
                new_path.append(Point(j - 1, t - 1, p_change.exp().item()))
                next_beams.append(
                    BeamState(token_index=j - 1, time_index=t - 1, score=change_score, path=new_path)
                )

        beams = sorted(next_beams, key=lambda x: x.score, reverse=True)[:beam_width]
        if not beams:
            break

    if not beams:
        return None

    best_beam = beams[0]
    t = best_beam.time_index
    j = best_beam.token_index
    while t > 0:
        prob = emission[t - 1, blank_id].exp().item()
        best_beam.path.append(Point(j, t - 1, prob))
        t -= 1

    return best_beam.path[::-1]


@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    @property
    def length(self) -> int:
        return self.end - self.start


def merge_repeats(path: List[Point], transcript: str) -> List[Segment]:
    i1, i2 = 0, 0
    segments: List[Segment] = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / max(i2 - i1, 1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments


__all__ = [
    "AlignedTranscriptionResult",
    "LANGUAGES_WITHOUT_SPACES",
    "SAMPLE_RATE",
    "SingleSegment",
    "load_align_model",
    "align",
    "load_audio",
]
