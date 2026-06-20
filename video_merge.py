"""
Video Merger - FFmpeg based
Concatenates clips, adds voice + music, exports final video
"""
import os
import subprocess
import json

OUTPUT_DIR = "outputs"
TEMP_DIR = "temp"


class VideoMerger:
    def __init__(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def merge(
        self,
        clips: list,
        voice_path: str,
        music_path: str,
        job_id: str,
        title: str = "video"
    ) -> str:
        """
        Merge all clips + audio into final video.
        Returns path to final MP4.
        """
        print(f"[Merger] Склеюю {len(clips)} кліпів...")

        # 1. Concatenate video clips
        concat_path = f"{TEMP_DIR}/{job_id}_concat.mp4"
        self._concat_clips(clips, concat_path)

        # 2. Mix voice + music
        mixed_audio = f"{TEMP_DIR}/{job_id}_audio_mix.aac"
        self._mix_audio(voice_path, music_path, mixed_audio)

        # 3. Combine video + audio
        safe_title = "".join(c for c in title if c.isalnum() or c in " _-")[:50]
        final_path = f"{OUTPUT_DIR}/{job_id}_{safe_title}.mp4"
        self._combine(concat_path, mixed_audio, final_path)

        # 4. Cleanup temp files
        self._cleanup([concat_path, mixed_audio] + clips)

        print(f"[Merger] ✅ Фінальне відео: {final_path}")
        return final_path

    def _concat_clips(self, clips: list, output: str):
        """Concatenate video clips using FFmpeg concat filter"""
        # Write concat list — path derived from output to avoid concurrent-job collisions
        list_path = output.replace("_concat.mp4", "_concat_list.txt")
        with open(list_path, "w") as f:
            for clip in clips:
                f.write(f"file '{os.path.abspath(clip)}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", list_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-an",          # no audio from clips
            output
        ]
        self._run(cmd, "concat clips")

    def _mix_audio(self, voice_path: str, music_path: str, output: str):
        """Mix voice (loud) + music (quiet background)"""
        cmd = [
            "ffmpeg", "-y",
            "-i", voice_path,
            "-i", music_path,
            "-filter_complex",
            # Voice at full volume, music at 15% in background
            "[0:a]volume=1.0[voice];"
            "[1:a]volume=0.15[music];"
            "[voice][music]amix=inputs=2:duration=first:dropout_transition=3[out]",
            "-map", "[out]",
            "-c:a", "aac",
            "-b:a", "192k",
            output
        ]
        self._run(cmd, "mix audio")

    def _combine(self, video_path: str, audio_path: str, output: str):
        """Combine video + mixed audio into final file"""
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",     # end when shortest stream ends
            "-movflags", "+faststart",  # optimize for web streaming
            output
        ]
        self._run(cmd, "combine video+audio")

    def _run(self, cmd: list, step: str):
        """Run FFmpeg command"""
        print(f"[Merger] FFmpeg: {step}...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"[Merger] FFmpeg stderr: {result.stderr[-500:]}")
            raise RuntimeError(f"FFmpeg failed at '{step}': {result.stderr[-200:]}")

    def _cleanup(self, paths: list):
        """Remove temporary files"""
        for path in paths:
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass
