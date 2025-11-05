## Aligned TTS

Python package that bundles CosyVoice2 synthesis with a wav2vec2-based forced aligner.
Given a text prompt, it generates the spoken audio and character-level timestamps.

### Installation

```bash
conda create -n cosyvoice python=3.10 -y
conda activate cosyvoice

# CosyVoice requirements
cd CosyVoice
pip install -r requirements.txt
mkdir -p pretrained_models
git clone https://www.modelscope.cn/iic/CosyVoice2-0.5B.git pretrained_models/CosyVoice2-0.5B

# Editable install for development
cd ..
pip install -e .
```

### Usage

```python
from aligned_tts import AlignedTTS

tts = AlignedTTS()

result = tts.synthesize("Which people make <strong>these</strong> words?", # use <strong></strong> to do emphasis
                        prompt_wav="reference_speech/adult_female.wav", # use prompt_wav to change speaker
                        manual_seed=200, # Sometimes you need to try several random seeds to achieve the best emphasis.
                        instruct_text="Speak slowly and with emphasis." # specail instruction for the model
                        )
print("Audio:", result.audio_path)
print("Alignment:", result.alignment_path)
```

```bash
# generate example video
./generate_mp4_from_srt.sh
```

Outputs are written to `generated_speech/` by default (`.wav`, `.json`, `.srt`).
Set `output_dir=` to customise names and paths.
