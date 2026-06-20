"""
Audio Generator - Kokoro TTS (voice) + MusicGen (background music)
"""
import os
import torch
import numpy as np
import soundfile as sf

OUTPUT_DIR = "temp"
VOICE_LANG  = os.environ.get("VOICE_LANG",  "uk")       # Ukrainian
VOICE_VOICE = os.environ.get("VOICE_VOICE", "uk_UA_Female")  # Ukrainian TTS voice


class AudioGenerator:
    def __init__(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self.tts_model = None
        self.music_model = None

    # ── TTS ───────────────────────────────────────────────────────────────────

    def _load_tts(self):
        if self.tts_model is not None:
            return
        print("[Audio] Завантажую Kokoro TTS...")
        from kokoro import KPipeline
        # 'uk' for Ukrainian, 'en-us' for English
        self.tts_model = KPipeline(lang_code=VOICE_LANG)
        print("[Audio] Kokoro TTS готовий ✅")

    def generate_voice(self, text: str, job_id: str) -> str:
        """Generate voice narration from text. Returns path to WAV file."""
        self._load_tts()

        output_path = f"{OUTPUT_DIR}/{job_id}_voice.wav"
        print(f"[Audio] Генерую голос ({len(text)} символів)...")

        # Kokoro splits long text automatically
        generator = self.tts_model(
            text,
            voice=VOICE_VOICE,
            speed=0.95,
            split_pattern=r'\n+'
        )

        all_audio = []
        sample_rate = 24000

        for i, (gs, ps, audio) in enumerate(generator):
            all_audio.append(audio)

        if all_audio:
            combined = np.concatenate(all_audio)
            sf.write(output_path, combined, sample_rate)
            print(f"[Audio] ✅ Голос збережено: {output_path}")
        else:
            # Fallback: silent audio
            silence = np.zeros(sample_rate * 5, dtype=np.float32)
            sf.write(output_path, silence, sample_rate)

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
