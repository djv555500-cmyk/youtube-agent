"""
YouTube Video Agent — Flask server
"""
import os, uuid, threading
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from apscheduler.schedulers.background import BackgroundScheduler

# Load .env
_env = Path(__file__).parent / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from database import JobDB
from agent import VideoAgent

app       = Flask(__name__, template_folder="templates")
db        = JobDB()
scheduler = BackgroundScheduler()
scheduler.start()

# Ensure output dirs exist
Path("outputs").mkdir(exist_ok=True)
Path("temp").mkdir(exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_agent(job_id: str):
    agent = VideoAgent()
    agent.run(job_id)


def _schedule_job(job_id: str, schedule: dict):
    stype = schedule.get("type", "now")
    if stype == "now":
        t = threading.Thread(target=_run_agent, args=(job_id,), daemon=True)
        t.start()
    elif stype == "once":
        run_date = schedule.get("run_date")
        scheduler.add_job(_run_agent, "date", run_date=run_date,
                          args=[job_id], id=job_id)
    elif stype == "daily":
        hour, minute = schedule.get("time", "08:00").split(":")
        scheduler.add_job(_run_agent, "cron", hour=int(hour), minute=int(minute),
                          args=[job_id], id=job_id)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/jobs", methods=["GET"])
def list_jobs():
    return jsonify(db.get_all_jobs())


@app.route("/api/jobs/<job_id>", methods=["GET"])
def get_job(job_id):
    job = db.get_job(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404
    return jsonify(job)


@app.route("/api/jobs", methods=["POST"])
def create_job():
    data     = request.json or {}
    job_id   = str(uuid.uuid4())[:8]
    config   = {
        "topic":            data.get("topic", ""),
        "resolution":       data.get("resolution", "720p"),
        "style":            data.get("style", "cinematic"),
        "duration":         data.get("duration", 5),
        "video_model":      data.get("video_model", "wan22"),
        "prompt_mode":      data.get("prompt_mode", "auto"),
        "manual_prompt":    data.get("manual_prompt", ""),
        "photo_mode":       data.get("photo_mode", "none"),
        "photo_approval":   data.get("photo_approval", False),
        "uploaded_photos":  data.get("uploaded_photos", []),
        "frame_continuity": data.get("frame_continuity", False),
        "clip_duration":    data.get("clip_duration", "5"),
        "fps":              data.get("fps", 16),
        "steps":            data.get("steps", 15),
        "voice_sample":     data.get("voice_sample", ""),
    }
    db.create_job(job_id, config)
    _schedule_job(job_id, data.get("schedule", {"type": "now"}))
    return jsonify({"job_id": job_id}), 201


@app.route("/api/jobs/<job_id>/cancel", methods=["POST"])
def cancel_job(job_id):
    db.update_job(job_id, status="cancelled")
    try:
        scheduler.remove_job(job_id)
    except Exception:
        pass
    return jsonify({"ok": True})


@app.route("/api/jobs/<job_id>/approve", methods=["POST"])
def approve_photo(job_id):
    data     = request.json or {}
    scene_id = data.get("scene_id")
    decision = data.get("decision", "approved")
    db.set_approval(job_id, scene_id, decision)
    return jsonify({"ok": True})


@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory("outputs", filename)


@app.route("/temp/<path:filename>")
def serve_temp(filename):
    return send_from_directory("temp", filename)


@app.route("/api/generate", methods=["POST"])
def generate_multi():
    """Bulk create jobs from UI (supports multiple topics + count)."""
    data     = request.json or {}
    topics_raw = data.get("topics", "")
    topics   = [t.strip() for t in topics_raw.split("\n") if t.strip()] or [topics_raw]
    count    = max(1, int(data.get("count", 1)))
    schedule_type = data.get("schedule", "now")
    sched_time    = data.get("time", "08:00")

    created = []
    for topic in topics:
        for _ in range(count):
            job_id = str(uuid.uuid4())[:8]
            config = {
                "topic":            topic,
                "resolution":       data.get("resolution", "720p"),
                "style":            data.get("style", "cinematic"),
                "duration":         data.get("duration", 2),
                "video_model":      data.get("video_model", "wan22"),
                "prompt_mode":      data.get("prompt_mode", "auto"),
                "manual_prompt":    data.get("manual_prompt", ""),
                "photo_mode":       data.get("photo_mode", "none"),
                "photo_approval":   data.get("photo_approval", False),
                "uploaded_photos":  data.get("uploaded_photos", []),
                "frame_continuity": data.get("frame_continuity", False),
                "clip_duration":    data.get("clip_duration", "5"),
                "fps":              data.get("fps", 16),
                "steps":            data.get("steps", 15),
                "voice_sample":     data.get("voice_sample", ""),
            }
            db.create_job(job_id, config)
            schedule = {"type": schedule_type}
            if schedule_type == "once":
                from datetime import date
                schedule["run_date"] = f"{date.today()}T{sched_time}:00"
            elif schedule_type == "daily":
                schedule["time"] = sched_time
            _schedule_job(job_id, schedule)
            created.append(job_id)

    return jsonify({"job_ids": created}), 201


@app.route("/api/job/<job_id>/cancel", methods=["POST"])
def cancel_job_alias(job_id):
    return cancel_job(job_id)


@app.route("/api/job/<job_id>/approve_photo", methods=["POST"])
def approve_photo_alias(job_id):
    data     = request.json or {}
    scene_id = data.get("scene_id")
    db.set_approval(job_id, scene_id, "approved")
    return jsonify({"ok": True})


@app.route("/api/job/<job_id>/reject_photo", methods=["POST"])
def reject_photo_alias(job_id):
    data     = request.json or {}
    scene_id = data.get("scene_id")
    db.set_approval(job_id, scene_id, "rejected")
    return jsonify({"ok": True})


@app.route("/api/upload_voice", methods=["POST"])
def upload_voice():
    """Save uploaded voice sample, return its path."""
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "no file"}), 400
    path = f"temp/voice_sample_{f.filename}"
    f.save(path)
    return jsonify({"path": path})


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[Agent] Running on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
