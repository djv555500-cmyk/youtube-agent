"""
Image Generator - FLUX for AI photo generation
Used when photo_mode = "generate"
"""
import os, torch
OUTPUT_DIR = "temp"

class ImageGenerator:
    def __init__(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self.pipe = None

    def _load(self):
        if self.pipe: return
        print("[ImageGen] Завантажую FLUX...")
        from diffusers import FluxPipeline
        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16
        ).to("cuda")
        print("[ImageGen] FLUX готовий ✅")

    def generate(self, prompt: str, scene_id: str, style: str = "cinematic") -> str:
        self._load()
        output_path = f"{OUTPUT_DIR}/{scene_id}_photo.png"

        style_suffixes = {
            "cinematic":    "cinematic photography, professional, dramatic lighting, 8K",
            "documentary":  "documentary photo, natural lighting, realistic, sharp",
            "animation":    "digital art, vivid colors, high detail, studio quality",
            "timelapse":    "professional photography, long exposure, high quality",
        }
        full_prompt = f"{prompt}, {style_suffixes.get(style, style_suffixes['cinematic'])}"

        print(f"[ImageGen] Генерую фото: {scene_id}")
        image = self.pipe(
            prompt=full_prompt,
            height=720, width=1280,
            num_inference_steps=4,      # FLUX schnell = fast
            guidance_scale=0.0,
        ).images[0]
        image.save(output_path)
        print(f"[ImageGen] ✅ Фото: {output_path}")
        return output_path

    def regenerate(self, prompt: str, scene_id: str, style: str = "cinematic") -> str:
        """Same as generate but with different seed"""
        import random
        self._load()
        output_path = f"{OUTPUT_DIR}/{scene_id}_photo.png"
        style_suffixes = {
            "cinematic":    "cinematic photography, professional, dramatic lighting, 8K",
            "documentary":  "documentary photo, natural lighting, realistic, sharp",
            "animation":    "digital art, vivid colors, high detail, studio quality",
            "timelapse":    "professional photography, long exposure, high quality",
        }
        full_prompt = f"{prompt}, {style_suffixes.get(style, style_suffixes['cinematic'])}"
        image = self.pipe(
            prompt=full_prompt,
            height=720, width=1280,
            num_inference_steps=4,
            guidance_scale=0.0,
            generator=torch.Generator(device='cuda').manual_seed(random.randint(0, 999999))
        ).images[0]
        image.save(output_path)
        return output_path
