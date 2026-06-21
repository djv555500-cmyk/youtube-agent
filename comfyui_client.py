"""
ComfyUI API client for WAN I2V generation.
"""
import json, uuid, time, urllib.request, urllib.parse, os
from PIL import Image

COMFY_URL = "http://127.0.0.1:8188"

GGUF_MODEL   = "wan2.1-i2v-14b-720p-Q8_0.gguf"
VAE_MODEL    = "Wan2.1_VAE.pth"
T5_MODEL     = "umt5_xxl_wan_i2v.safetensors"
CLIP_MODEL   = "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"

# Snap num_frames to WAN's required (length-1) % 4 == 0
def _snap_frames(n):
    remainder = (n - 1) % 4
    return n if remainder == 0 else n + (4 - remainder)


def _build_workflow(prompt, negative, width, height, num_frames, fps, steps,
                    image_path=None, cfg=6.0, output_prefix="comfy_out"):
    """Build ComfyUI WAN workflow dict (T2V or I2V via LoadImage)."""
    num_frames = _snap_frames(int(num_frames))
    seed = int(time.time()) % 2**32

    wf = {
        "1": {"class_type": "UnetLoaderGGUF",
              "inputs": {"unet_name": GGUF_MODEL}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": T5_MODEL, "type": "wan"}},
        "3": {"class_type": "VAELoader",
              "inputs": {"vae_name": VAE_MODEL}},
        "4": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["2", 0]}},
        "5": {"class_type": "CLIPTextEncode",
              "inputs": {"text": negative, "clip": ["2", 0]}},
        "6": {"class_type": "WanImageToVideo",
              "inputs": {
                  "positive": ["4", 0],
                  "negative": ["5", 0],
                  "vae": ["3", 0],
                  "width": width,
                  "height": height,
                  "length": num_frames,
                  "batch_size": 1,
              }},
        "7": {"class_type": "KSampler",
              "inputs": {
                  "model": ["1", 0],
                  "positive": ["6", 0],
                  "negative": ["6", 1],
                  "latent_image": ["6", 2],
                  "seed": seed,
                  "steps": steps,
                  "cfg": cfg,
                  "sampler_name": "euler",
                  "scheduler": "simple",
                  "denoise": 1.0,
              }},
        "8": {"class_type": "VAEDecode",
              "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "VHS_VideoCombine",
               "inputs": {
                   "images": ["8", 0],
                   "frame_rate": fps,
                   "loop_count": 0,
                   "filename_prefix": output_prefix,
                   "format": "video/h264-mp4",
                   "pix_fmt": "yuv420p",
                   "crf": 19,
                   "save_metadata": False,
                   "pingpong": False,
                   "save_output": True,
               }},
    }

    # I2V: add CLIP vision + start image via LoadImage (saves image to ComfyUI input dir)
    if image_path and os.path.exists(image_path):
        import shutil
        comfy_input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "ComfyUI", "input")
        os.makedirs(comfy_input_dir, exist_ok=True)
        dest_name = "i2v_start_" + os.path.basename(image_path)
        shutil.copy2(image_path, os.path.join(comfy_input_dir, dest_name))

        wf["10"] = {"class_type": "CLIPVisionLoader",
                    "inputs": {"clip_name": CLIP_MODEL}}
        wf["11"] = {"class_type": "LoadImage",
                    "inputs": {"image": dest_name}}
        wf["12"] = {"class_type": "CLIPVisionEncode",
                    "inputs": {"clip_vision": ["10", 0], "image": ["11", 0]}}
        wf["6"]["inputs"]["clip_vision_output"] = ["12", 0]
        wf["6"]["inputs"]["start_image"] = ["11", 0]

    return wf


def queue_prompt(workflow: dict) -> str:
    """Submit workflow and return prompt_id."""
    payload = json.dumps({"prompt": workflow, "client_id": str(uuid.uuid4())}).encode()
    req = urllib.request.Request(f"{COMFY_URL}/prompt",
                                 data=payload,
                                 headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req)
    return json.loads(resp.read())["prompt_id"]


def wait_for_completion(prompt_id: str, timeout: int = 900) -> dict:
    """Poll until prompt completes, return history entry."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = urllib.request.urlopen(f"{COMFY_URL}/history/{prompt_id}")
        hist = json.loads(resp.read())
        if prompt_id in hist:
            return hist[prompt_id]
        time.sleep(3)
    raise TimeoutError(f"ComfyUI prompt {prompt_id} timed out after {timeout}s")


def generate_video(prompt, output_path, duration=5, resolution="480p",
                   fps=16, steps=20, image_path=None, negative="") -> str:
    """
    Generate video via ComfyUI WAN and save to output_path.
    Returns output_path on success.
    """
    w, h = (1280, 720) if resolution in ("720p", "1080p") else (848, 480)
    num_frames = max(9, duration * fps)

    # Resize start image to target resolution if provided
    resized_path = None
    if image_path and os.path.exists(image_path):
        resized_path = image_path + "_resized.png"
        img = Image.open(image_path).convert("RGB").resize((w, h))
        img.save(resized_path)

    prefix = os.path.splitext(os.path.basename(output_path))[0]
    wf = _build_workflow(prompt, negative, w, h, int(num_frames), fps, steps,
                         image_path=resized_path, output_prefix=prefix)

    prompt_id = queue_prompt(wf)
    print(f"[ComfyUI] Queued {prompt_id} — waiting...")
    result = wait_for_completion(prompt_id)

    # Find output video file (VHS_VideoCombine returns under "gifs" or "videos")
    for node_id, node_out in result.get("outputs", {}).items():
        for key in ("gifs", "videos"):
            for entry in node_out.get(key, []):
                fname = entry.get("filename", "")
                src = os.path.join("/workspace/ComfyUI/output", fname)
                if fname and os.path.exists(src):
                    os.rename(src, output_path)
                    print(f"[ComfyUI] ✅ Video saved: {output_path}")
                    return output_path

    raise RuntimeError(f"No video output found. History status: {result.get('status')}")


if __name__ == "__main__":
    # Quick smoke test
    out = generate_video(
        prompt="A calm ocean wave at sunset, cinematic",
        output_path="/workspace/temp/test_comfy.mp4",
        duration=3, resolution="480p", fps=16, steps=15,
    )
    print("Done:", out)
