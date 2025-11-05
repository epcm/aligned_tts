from aligned_tts import AlignedTTS

if __name__ == "__main__":
    tts = AlignedTTS()
    result = tts.synthesize("Which people make <strong>these</strong> words?", 
                            prompt_wav="reference_speech/adult_female.wav", 
                            manual_seed=1024, 
                            instruct_text="Speak slowly and with emphasis.")
    print("Audio:", result.audio_path)
    print("Alignment:", result.alignment_path)
