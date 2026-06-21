"""
Video Generator - WAN 2.2 + LTX-Video
Supports:
  - text-to-video, image-to-video
  - frame continuity (last frame → first frame of next clip)
  - configurable clip duration: 5 | 10 | 15 | 20 | auto
"""
import os, torch, subprocess, shutil

def _ffmpeg():
    p = shutil.which("ffmpeg")
    if p: return p
    c = "/workspace/bin/ffmpeg"
    return c if os.path.exists(c) else "ffmpeg"
OUTPUT_DIR = "temp"

class VideoGenerator:
    # Class-level cache so models survive across multiple job instances
    _cls_wan_pipe  = None
    _cls_wan_t2v   = None
    _cls_wan_i2v   = None
    _cls_wan_export = None
    _cls_ltx_pipe  = None
    _cls_ltx_i2v   = None
    _cls_ltx_t2v   = None

    def __init__(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Public ────────────────────────────────────────────────────────────────

    def generate(self, prompt, duration=5, resolution="720p",
                 style="cinematic", scene_id="scene",
                 image_path=None, model="wan22",
                 fps=16, steps=15) -> str:
        """
        Generate a video clip.
        image_path: None = text-to-video | path = image-to-video (frame continuity)
        Returns: path to MP4
        """
        output_path = f"{OUTPUT_DIR}/{scene_id}.mp4"
        enhanced    = self._enhance_prompt(prompt, style)
        negative    = ("blurry, low quality, distorted, watermark, text overlay, "
                       "bad anatomy, artifacts, overexposed")
        w, h = (1280, 720) if resolution in ("720p", "1080p") else (848, 480)
        frames = max(9, duration * fps)  # LTX minimum 9 frames

        print(f"[VideoGen] {scene_id} | {model} | {resolution} | {duration}s | {fps}fps | {steps}steps | continuity={'yes' if image_path else 'no'}")

        try:
            if model == "ltx":
                return self._gen_ltx(enhanced, negative, w, h, frames, fps, image_path, output_path)
            elif model == "both":
                if duration >= 8:
                    return self._gen_wan(enhanced, negative, w, h, frames, fps, image_path, output_path, steps)
                else:
                    return self._gen_ltx(enhanced, negative, w, h, frames, fps, image_path, output_path)
            else:
                return self._gen_wan(enhanced, negative, w, h, frames, fps, image_path, output_path, steps)

        except torch.cuda.OutOfMemoryError:
            print("[VideoGen] ⚠️ OOM — зменшую роздільність...")
            torch.cuda.empty_cache()
            w, h = 848, 480
            if model == "ltx":
                return self._gen_ltx(enhanced, negative, w, h, frames, fps, image_path, output_path)
            return self._gen_wan(enhanced, negative, w, h, frames, fps, image_path, output_path, steps)

    def extract_last_frame(self, video_path: str, output_path: str) -> str:
        """
        Extract the very last frame of a video as a PNG image.
        Used for frame continuity: last frame → first frame of next clip.
        """
        cmd = [
            _ffmpeg(), "-y",
            "-sseof", "-0.1",       # seek to 0.1s before end
            "-i", video_path,
            "-frames:v", "1",
            "-q:v", "1",
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0 or not os.path.exists(output_path):
            print(f"[VideoGen] ⚠️ Не вдалось витягти останній кадр з {video_path}")
            return None
        print(f"[VideoGen] ✅ Останній кадр: {output_path}")
        return output_path

    WAN_T2V_PATH = "/workspace/models/wan_diffusers/models--Wan-AI--Wan2.1-T2V-14B-Diffusers/snapshots/38ec498cb3208fb688890f8cc7e94ede2cbd7f68"
    WAN_I2V_PATH = "/workspace/models/wan_i2v"

    # ── WAN 2.1 ───────────────────────────────────────────────────────────────

    @property
    def _wan_pipe(self):  return VideoGenerator._cls_wan_pipe
    @_wan_pipe.setter
    def _wan_pipe(self, v): VideoGenerator._cls_wan_pipe = v

    @property
    def _wan_t2v(self):  return VideoGenerator._cls_wan_t2v
    @_wan_t2v.setter
    def _wan_t2v(self, v): VideoGenerator._cls_wan_t2v = v

    @property
    def _wan_i2v(self):  return VideoGenerator._cls_wan_i2v
    @_wan_i2v.setter
    def _wan_i2v(self, v): VideoGenerator._cls_wan_i2v = v

    @property
    def _export(self):  return VideoGenerator._cls_wan_export
    @_export.setter
    def _export(self, v): VideoGenerator._cls_wan_export = v

    def _load_wan(self):
        if VideoGenerator._cls_wan_pipe: return
        from diffusers import WanPipeline, WanImageToVideoPipeline
        from diffusers.utils import export_to_video

        if os.path.exists(self.WAN_T2V_PATH):
            print("[VideoGen] Завантажую WAN 2.1 T2V 14B...")
            VideoGenerator._cls_wan_t2v = WanPipeline.from_pretrained(
                self.WAN_T2V_PATH,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True
            )
            VideoGenerator._cls_wan_t2v.enable_model_cpu_offload()
            VideoGenerator._cls_wan_t2v.enable_vae_slicing()
        else:
            VideoGenerator._cls_wan_t2v = None
            print("[VideoGen] ⚠️ WAN T2V не знайдено в", self.WAN_T2V_PATH)

        if os.path.exists(self.WAN_I2V_PATH):
            print("[VideoGen] Завантажую WAN I2V 14B...")
            VideoGenerator._cls_wan_i2v = WanImageToVideoPipeline.from_pretrained(
                self.WAN_I2V_PATH,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            VideoGenerator._cls_wan_i2v.enable_sequential_cpu_offload()
            if hasattr(VideoGenerator._cls_wan_i2v, "enable_vae_slicing"):
                VideoGenerator._cls_wan_i2v.enable_vae_slicing()
            print("[VideoGen] WAN I2V готовий (CPU offload) ✅")
        else:
            VideoGenerator._cls_wan_i2v = None
            print("[VideoGen] ⚠️ WAN I2V не знайдено в", self.WAN_I2V_PATH)

        VideoGenerator._cls_wan_export = export_to_video
        VideoGenerator._cls_wan_pipe   = True
        print("[VideoGen] WAN 2.1 14B готовий ✅")

    def _gen_wan(self, prompt, negative, w, h, frames, fps, image_path, output_path, steps=15):
        """Generate via ComfyUI if available, else fall back to diffusers."""
        if self._comfyui_available():
            return self._gen_wan_comfyui(prompt, negative, w, h, frames, fps,
                                          image_path, output_path, steps)
        return self._gen_wan_diffusers(prompt, negative, w, h, frames, fps,
                                        image_path, output_path, steps)

    def _comfyui_available(self):
        try:
            import urllib.request
            urllib.request.urlopen("http://127.0.0.1:8188/system_stats", timeout=2)
            return True
        except Exception:
            return False

    def _gen_wan_comfyui(self, prompt, negative, w, h, frames, fps, image_path, output_path, steps):
        from comfyui_client import generate_video
        resolution = "720p" if h >= 720 else "480p"
        duration = frames / fps
        print(f"[VideoGen] ComfyUI WAN | {resolution} | {duration:.1f}s | {steps}steps")
        return generate_video(prompt, output_path, duration=duration, resolution=resolution,
                               fps=fps, steps=steps, image_path=image_path, negative=negative)

    def _gen_wan_diffusers(self, prompt, negative, w, h, frames, fps, image_path, output_path, steps=15):
        self._load_wan()
        from PIL import Image
        import numpy as np

        use_i2v = image_path and os.path.exists(image_path)

        if not use_i2v and self._wan_t2v is None and self._wan_i2v is not None:
            gray = Image.fromarray(np.full((h, w, 3), 128, dtype=np.uint8))
            tmp_path = output_path.replace(".mp4", "_init.png")
            gray.save(tmp_path)
            image_path = tmp_path
            use_i2v = True
            print("[VideoGen] T2V недоступна — використовую I2V з нейтральним кадром")

        if use_i2v:
            if self._wan_i2v is None:
                raise RuntimeError(
                    f"WAN I2V модель не знайдена в {self.WAN_I2V_PATH}. "
                    "Завантажте: huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P-Diffusers "
                    f"--local-dir {self.WAN_I2V_PATH}"
                )
            img = Image.open(image_path).convert("RGB").resize((w, h))
            out = self._wan_i2v(
                image=img, prompt=prompt, negative_prompt=negative,
                height=h, width=w, num_frames=frames,
                guidance_scale=5.0, num_inference_steps=steps,
            ).frames[0]
        else:
            out = self._wan_t2v(
                prompt=prompt, negative_prompt=negative,
                height=h, width=w, num_frames=frames,
                guidance_scale=5.0, num_inference_steps=30,
            ).frames[0]
        self._export(out, output_path, fps=fps)
        return output_path

    # ── LTX-Video ─────────────────────────────────────────────────────────────

    def _load_ltx(self):
        if self._ltx_pipe: return
        print("[VideoGen] Завантажую LTX-Video...")
        from diffusers import LTXImageToVideoPipeline, LTXPipeline
        from diffusers.utils import export_to_video
        self._ltx_i2v = LTXImageToVideoPipeline.from_pretrained(
            "Lightricks/LTX-Video", torch_dtype=torch.bfloat16).to("cuda")
        self._ltx_t2v = LTXPipeline.from_pretrained(
            "Lightricks/LTX-Video", torch_dtype=torch.bfloat16).to("cuda")
        self._export  = export_to_video
        self._ltx_pipe = True
        print("[VideoGen] LTX-Video готовий ✅")

    def _gen_ltx(self, prompt, negative, w, h, frames, fps, image_path, output_path):
        self._load_ltx()
        from PIL import Image
        # LTX requires frames = N*8+1
        frames = ((frames - 1) // 8) * 8 + 1
        if image_path and os.path.exists(image_path):
            img = Image.open(image_path).convert("RGB").resize((w, h))
            out = self._ltx_i2v(
                image=img, prompt=prompt, negative_prompt=negative,
                height=h, width=w, num_frames=frames,
                num_inference_steps=25, guidance_scale=3.0,
            ).frames[0]
        else:
            out = self._ltx_t2v(
                prompt=prompt, negative_prompt=negative,
                height=h, width=w, num_frames=frames,
                num_inference_steps=25, guidance_scale=3.0,
            ).frames[0]
        self._export(out, output_path, fps=fps)
        return output_path

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _enhance_prompt(self, prompt, style):
        suffixes = {
            "cinematic":   "cinematic shot, professional cinematography, dramatic lighting, 8K",
            "documentary": "documentary style, natural lighting, realistic, professional camera",
            "animation":   "high quality animation, smooth motion, vivid colors, studio quality",
            "timelapse":   "timelapse photography, smooth motion, high quality",
        }
        return f"{prompt}, {suffixes.get(style, suffixes['cinematic'])}"
