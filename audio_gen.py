"""
Audio Generator - XTTS v2 (voice cloning) + MusicGen (background music)
"""
import os
import torch
import numpy as np
import soundfile as sf

OUTPUT_DIR  = "temp"
VOICE_LANG  = os.environ.get("VOICE_LANG", "uk")
# Path to a WAV/MP3 sample of the voice to clone (3-10 sec)
VOICE_SAMPLE = os.environ.get("VOICE_SAMPLE", "")


class AudioGenerator:
    def __init__(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self.tts_model = None
        self.music_model = None

    # ── TTS ───────────────────────────────────────────────────────────────────

    def _load_tts(self):
        if self.tts_model is not None:
            return
        import os as _os
        _os.environ["COQUI_TOS_AGREED"] = "1"
        print("[Audio] Завантажую XTTS v2...")
        from TTS.api import TTS
        use_gpu = torch.cuda.is_available() and not self._gpu_busy()
        self.tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=use_gpu)
        print(f"[Audio] XTTS v2 готовий ✅ (gpu={use_gpu})")

    def _gpu_busy(self):
        try:
            free = torch.cuda.mem_get_info()[0] / 1024**3
            return free < 4.0  # less than 4GB free → use CPU
        except Exception:
            return True

    def generate_voice(self, text: str, job_id: str) -> str:
        """Generate voice narration. Uses voice cloning if VOICE_SAMPLE is set."""
        self._load_tts()

        output_path = f"{OUTPUT_DIR}/{job_id}_voice.wav"
        print(f"[Audio] Генерую голос ({len(text)} символів)...")

        speaker_wav = VOICE_SAMPLE if VOICE_SAMPLE and os.path.exists(VOICE_SAMPLE) else None

        if speaker_wav:
            print(f"[Audio] Клонування голосу з: {speaker_wav}")
            self.tts_model.tts_to_file(
                text=text,
                speaker_wav=speaker_wav,
                language=VOICE_LANG,
                file_path=output_path,
            )
        else:
            # Fallback: built-in speaker
            speakers = self.tts_model.speakers or []
            speaker = next((s for s in speakers if "uk" in s.lower()), None) or (speakers[0] if speakers else None)
            self.tts_model.tts_to_file(
                text=text,
                speaker=speaker,
                language=VOICE_LANG,
                file_path=output_path,
            )

        print(f"[Audio] ✅ Голос збережено: {output_path}")
        return output_path

    # ── Music ─────────────────────────────────────────────────────────────────

    def _load_music(self):
        if self.music_model is not None:
            return
        print("[Audio] Завантажую MusicGen...")
        from audiocraft.models import MusicGen
        self.music_model = MusicGen.get_pretrained("facebook/musicgen-medium")
        self.music_model.set_generation_params(
            use_sampling=True,
            top_k=250,
            duration=30
        )
        print("[Audio] MusicGen готовий ✅")

    def generate_music(self, mood: str, duration_seconds: int, job_id: str) -> str:
        """Generate background music. Returns path to WAV file."""
        self._load_music()

        output_path = f"{OUTPUT_DIR}/{job_id}_music.wav"
        print(f"[Audio] Генерую музику: {mood}, {duration_seconds}s...")

        prompts = {
            "dramatic":    "dramatic cinematic orchestral music, powerful, emotional, Hans Zimmer style",
            "calm":        "calm ambient background music, peaceful, gentle piano and strings",
            "mysterious":  "mysterious atmospheric music, dark ambient, cinematic tension",
            "epic":        "epic orchestral music, heroic, powerful, movie soundtrack",
            "uplifting":   "uplifting inspirational background music, positive, motivational",
            "tense":       "tense suspenseful background music, thriller style, building tension",
            "cinematic":   "cinematic background music, neutral, professional documentary style",
        }

        prompt = prompts.get(mood, prompts["cinematic"])

        # Generate in chunks if needed (MusicGen max ~30s at once)
        chunk_duration = 28
        chunks = []
        remaining = duration_seconds

        self.music_model.set_generation_params(
            use_sampling=True,
            top_k=250,
            duration=min(chunk_duration, remaining)
        )

        while remaining > 0:
            dur = min(chunk_duration, remaining)
            self.music_model.set_generation_params(
                use_sampling=True, top_k=250, duration=dur
            )
            wav = self.music_model.generate([prompt])
            audio_np = wav[0, 0].cpu().numpy()
            chunks.append(audio_np)
            remaining -= dur

        combined = np.concatenate(chunks)
        sample_rate = self.music_model.sample_rate
        sf.write(output_path, combined, sample_rate)

        print(f"[Audio] ✅ Музика збережена: {output_path}")
        return output_path
