"""
Video Agent v3
New features:
  - frame_continuity: last frame of clip N → first frame of clip N+1
  - clip_duration: 5 | 10 | 15 | 20 | auto
"""
import os, json, base64, anthropic
from pathlib import Path

# Load .env
_env = Path(__file__).parent / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
from database import JobDB
from video_gen import VideoGenerator
from image_gen import ImageGenerator
from audio_gen import AudioGenerator
from video_merge import VideoMerger

db     = JobDB()
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

SYSTEM_PROMPT = """Ти — режисер YouTube відео. Створюєш детальний план по темі.

Повертай ТІЛЬКИ JSON:
{
  "title": "Назва відео",
  "narration_intro": "Вступний текст українською",
  "narration_outro": "Завершальний текст українською",
  "scenes": [
    {
      "id": 1,
      "duration_seconds": 5,
      "video_prompt": "Detailed English prompt with lighting, camera, mood, movement.",
      "image_prompt": "Detailed English prompt for a STILL IMAGE to be animated.",
      "narration": "Текст українською для озвучення цієї сцени",
      "music_mood": "dramatic|calm|mysterious|epic|uplifting|tense",
      "style_hint": "cinematic|documentary|animation|timelapse",
      "is_key_scene": true
    }
  ]
}

Правила:
- Масштабуй кількість сцен до тривалості відео
- video_prompt і image_prompt ЗАВЖДИ англійською, максимально деталізовано
- narration українською, природній розмовний стиль
- Перша сцена завжди динамічна і захоплива
- is_key_scene: true для 25% найважливіших сцен
- Якщо clip_duration=auto: встановлюй duration_seconds залежно від типу сцени
  (екшн=5, пейзаж=15, стандарт=10)"""


class VideoAgent:
    def __init__(self):
        self.video_gen = VideoGenerator()
        self.image_gen = ImageGenerator()
        self.audio_gen = AudioGenerator()
        self.merger    = VideoMerger()

    def run(self, job_id: str):
        job = db.get_job(job_id)
        if not job: return
        cfg = job["config"]

        topic            = cfg["topic"]
        resolution       = cfg.get("resolution", "720p")
        style            = cfg.get("style", "cinematic")
        video_model      = cfg.get("video_model", "wan22")
        prompt_mode      = cfg.get("prompt_mode", "auto")
        manual_prompt    = cfg.get("manual_prompt", "")
        photo_mode       = cfg.get("photo_mode", "none")
        photo_approval   = cfg.get("photo_approval", False)
        uploaded_photos  = cfg.get("uploaded_photos", [])
        frame_continuity = cfg.get("frame_continuity", False)
        clip_duration    = cfg.get("clip_duration", "5")   # "5"|"10"|"15"|"20"|"auto"
        fps              = int(cfg.get("fps", 16))
        steps            = int(cfg.get("steps", 15))
        voice_sample     = cfg.get("voice_sample", "")

        try:
            # ── Step 1: Plan ──────────────────────────────────────────────────
            db.update_job(job_id, status="running", progress=5,
                         progress_text="🧠 Claude планує відео...")

            if prompt_mode == "manual":
                plan = self._manual_plan(manual_prompt, topic, clip_duration)
            else:
                plan = self._auto_plan(topic, style, cfg.get("duration", 5),
                                       clip_duration, job_id)

            scenes = plan["scenes"]
            total  = len(scenes)
            db.add_log(job_id, f"✅ План: {total} сцен — «{plan.get('title', topic)}»")

            saved_uploads = self._save_uploads(uploaded_photos, job_id)

            # ── Step 2: Generate clips ────────────────────────────────────────
            clip_paths   = []
            last_frame   = None   # for frame continuity

            for i, scene in enumerate(scenes):
                # ── Check if cancelled ────────────────────────────────────────
                current = db.get_job(job_id)
                if current and current.get("status") == "cancelled":
                    db.add_log(job_id, "⛔ Генерацію скасовано користувачем")
                    return

                progress = 10 + int((i / total) * 60)
                scene_id = f"{job_id}_s{i+1:03d}"
                dur      = self._resolve_duration(scene, clip_duration)

                # ── Determine image source ────────────────────────────────────
                image_path = None

                if frame_continuity and last_frame:
                    # Use last frame of previous clip as first frame
                    image_path = last_frame
                    db.add_log(job_id, f"🔗 Сцена {i+1}: continuity з попереднього кліпу")

                elif photo_mode == "upload" and saved_uploads:
                    image_path = saved_uploads[i % len(saved_uploads)]
                    db.update_job(job_id, progress=progress,
                        progress_text=f"🖼 Сцена {i+1}/{total}: використовую ваше фото...")

                elif photo_mode == "generate":
                    db.update_job(job_id, progress=progress,
                        progress_text=f"🎨 Сцена {i+1}/{total}: генерую фото (FLUX)...")
                    db.add_log(job_id, f"FLUX генерує фото для сцени {i+1}...")

                    image_path = self.image_gen.generate(
                        prompt=scene.get("image_prompt", scene["video_prompt"]),
                        scene_id=scene_id, style=style
                    )

                    if photo_approval:
                        db.set_pending_approval(job_id, scene_id, image_path)
                        db.update_job(job_id,
                            progress_text=f"⏸ Чекаю підтвердження фото {i+1}/{total}...")
                        db.add_log(job_id, f"⏸ Очікую підтвердження фото сцени {i+1}")

                        decision = db.wait_for_approval(job_id, scene_id, timeout=600)
                        if decision == "rejected":
                            db.add_log(job_id, f"🔄 Перегенеровую фото сцени {i+1}...")
                            image_path = self.image_gen.regenerate(
                                prompt=scene.get("image_prompt", scene["video_prompt"]),
                                scene_id=scene_id, style=style
                            )

                # ── Generate video clip ───────────────────────────────────────
                db.update_job(job_id, progress=progress,
                    progress_text=f"🎬 Відео кліп {i+1}/{total} ({dur}сек)...")
                db.add_log(job_id, f"Кліп {i+1} [{dur}s]: {scene['video_prompt'][:55]}...")

                clip = self.video_gen.generate(
                    prompt=scene["video_prompt"],
                    duration=dur,
                    resolution=resolution,
                    style=scene.get("style_hint", style),
                    scene_id=scene_id,
                    image_path=image_path,
                    model=video_model,
                    fps=fps,
                    steps=steps,
                )
                clip_paths.append(clip)

                # ── Extract last frame for next clip (if continuity on) ───────
                if frame_continuity:
                    frame_path = f"temp/{scene_id}_lastframe.png"
                    extracted  = self.video_gen.extract_last_frame(clip, frame_path)
                    last_frame = extracted  # None if failed → next clip uses text-to-video

            # ── Step 3: Voice ─────────────────────────────────────────────────
            db.update_job(job_id, progress=72, progress_text="🎙️ Генерація голосу...")
            narration  = self._build_narration(plan, scenes)
            if voice_sample:
                import os as _os
                _os.environ["VOICE_SAMPLE"] = voice_sample
            voice_path = self.audio_gen.generate_voice(narration, job_id)

            # ── Step 4: Music (optional — skip if audiocraft not installed) ────
            music_path = None
            try:
                db.update_job(job_id, progress=82, progress_text="🎵 Генерація музики...")
                mood       = self._dominant_mood(scenes)
                total_dur  = sum(self._resolve_duration(s, clip_duration) for s in scenes) + 10
                music_path = self.audio_gen.generate_music(mood, total_dur, job_id)
            except Exception as me:
                db.add_log(job_id, f"⚠️ Музика пропущена: {type(me).__name__}")

            # ── Step 5: Merge ─────────────────────────────────────────────────
            db.update_job(job_id, progress=92, progress_text="✂️ Склеюю фінальне відео...")
            output = self.merger.merge(
                clips=clip_paths, voice_path=voice_path,
                music_path=music_path, job_id=job_id,
                title=plan.get("title", topic)
            )

            db.update_job(job_id, status="done", progress=100,
                         progress_text="✅ Готово!", output_path=output)
            db.add_log(job_id, f"✅ Збережено: {output}")

        except Exception as e:
            db.update_job(job_id, status="error", progress_text=f"❌ {str(e)}")
            db.add_log(job_id, f"❌ Помилка: {str(e)}")
            raise

    # ── Planning ──────────────────────────────────────────────────────────────

    def _auto_plan(self, topic, style, duration_min, clip_duration, job_id):
        # Estimate scene count
        if clip_duration == "auto":
            avg_dur = 10
        else:
            avg_dur = int(clip_duration)
        scene_count = max(1, int((duration_min * 60) / avg_dur))

        msg = (f'Тема: "{topic}"\nСтиль: {style}\n'
               f'Тривалість відео: ~{duration_min} хв (~{scene_count} сцен)\n'
               f'Тривалість кліпу: {clip_duration} сек\n'
               f'Мова озвучення: українська')

        resp = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=8192,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": msg}]
        )
        text = resp.content[0].text.strip()
        import re
        m = re.search(r'```(?:json)?\s*([\s\S]+?)```', text)
        if m:
            text = m.group(1).strip()
        return json.loads(text)

    def _manual_plan(self, manual_prompt, topic, clip_duration):
        dur = 10 if clip_duration == "auto" else int(clip_duration)
        paragraphs = [p.strip() for p in manual_prompt.split("\n") if p.strip()] or [manual_prompt]
        scenes = [{
            "id": i + 1,
            "duration_seconds": dur,
            "video_prompt": p,
            "image_prompt": p,
            "narration": p,
            "music_mood": "cinematic",
            "style_hint": "cinematic",
            "is_key_scene": i == 0
        } for i, p in enumerate(paragraphs)]
        return {"title": topic, "scenes": scenes,
                "narration_intro": "", "narration_outro": ""}

    def _resolve_duration(self, scene: dict, clip_duration: str) -> int:
        """Return actual clip duration in seconds"""
        if clip_duration == "auto":
            return int(scene.get("duration_seconds", 10))
        return int(clip_duration)

    def _build_narration(self, plan, scenes):
        parts = []
        if plan.get("narration_intro"): parts.append(plan["narration_intro"])
        for s in scenes:
            if s.get("narration"): parts.append(s["narration"])
        if plan.get("narration_outro"): parts.append(plan["narration_outro"])
        return " ".join(parts)

    def _dominant_mood(self, scenes):
        moods = [s.get("music_mood", "cinematic") for s in scenes]
        return max(set(moods), key=moods.count)

    def _save_uploads(self, uploads_b64, job_id):
        paths = []
        for i, b64 in enumerate(uploads_b64):
            try:
                data = base64.b64decode(b64.split(",")[-1])
                path = f"temp/{job_id}_upload_{i}.jpg"
                with open(path, "wb") as f: f.write(data)
                paths.append(path)
            except Exception as e:
                print(f"[Agent] Помилка збереження фото {i}: {e}")
        return paths
