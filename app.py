"""
YouTube Video Agent - Main Flask Application v3
- Multiple topics support
- LTX fix
"""
import os, json, threading, uuid
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from agent import VideoAgent
from database import JobDB

app = Flask(__name__)
scheduler = BackgroundScheduler()
scheduler.start()
db = JobDB()
agent = VideoAgent()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/generate", methods=["POST"])
def generate():
    data         = request.json
    topics_raw   = data.get("topics", "").strip()
    schedule     = data.get("schedule", "now")
    time_str     = data.get("time", "")
    resolution   = data.get("resolution", "720p")
    duration     = int(data.get("duration", 5))
    style        = data.get("style", "cinematic")
    video_model  = data.get("video_model", "wan22")
    prompt_mode  = data.get("prompt_mode", "auto")
    manual_prompt= data.get("manual_prompt", "").strip()
    photo_mode   = data.get("photo_mode", "none")
    photo_approval = data.get("photo_approval", False)
    uploaded_photos = data.get("uploaded_photos", [])
    clip_duration= data.get("clip_duration", "5")
    frame_continuity = data.get("frame_continuity", False)

    # Parse multiple topics (one per line)
    topics = [t.strip() for t in topics_raw.splitlines() if t.strip()]
    if not topics:
        return jsonify({"error": "Введи хоча б одну тему"}), 400

    jobs = []
    for topic in topics:
        job_id = str(uuid.uuid4())[:8]
        config = dict(
            topic=topic, resolution=resolution, duration=duration,
            style=style, video_model=video_model,
            prompt_mode=prompt_mode, manual_prompt=manual_prompt,
            photo_mode=photo_mode, photo_approval=photo_approval,
            uploaded_photos=uploaded_photos,
            clip_duration=clip_duration,
            frame_continuity=frame_continuity
        )
        db.create_job(job_id, config)
        jobs.append(job_id)

    if schedule == "now":
        for job_id in jobs:
            threading.Thread(target=agent.run, args=(job_id,), daemon=True).start()
        return jsonify({"status": "started", "jobs": jobs, "count": len(jobs)})

    elif schedule == "once" and time_str:
        hour, minute = map(int, time_str.split(":"))
        for job_id in jobs:
            scheduler.add_job(agent.run, "date",
                run_date=_next_time(hour, minute), args=[job_id], id=job_id)
        return jsonify({"status": "scheduled", "jobs": jobs, "time": time_str})

    elif schedule == "daily" and time_str:
        hour, minute = map(int, time_str.split(":"))
        for job_id in jobs:
            scheduler.add_job(agent.run, CronTrigger(hour=hour, minute=minute),
                args=[job_id], id=f"daily_{job_id}")
        return jsonify({"status": "daily_set", "jobs": jobs, "time": time_str})

    return jsonify({"error": "Невірні параметри"}), 400

@app.route("/api/job/<job_id>/pending_approval")
def pending_approval(job_id):
    return jsonify(db.get_pending_approval(job_id))

@app.route("/api/job/<job_id>/approve_photo", methods=["POST"])
def approve_photo(job_id):
    data = request.json
    db.set_approval(job_id, data.get("scene_id"), "approved")
    return jsonify({"status": "approved"})

@app.route("/api/job/<job_id>/reject_photo", methods=["POST"])
def reject_photo(job_id):
    data = request.json
    db.set_approval(job_id, data.get("scene_id"), "rejected")
    return jsonify({"status": "rejected"})

@app.route("/api/jobs")
def get_jobs():
    return jsonify(db.get_all_jobs())

@app.route("/api/job/<job_id>")
def get_job(job_id):
    return jsonify(db.get_job(job_id))

@app.route("/api/job/<job_id>/cancel", methods=["POST"])
def cancel_job(job_id):
    db.update_job(job_id, status="cancelled")
    try: scheduler.remove_job(job_id)
    except: pass
    return jsonify({"status": "cancelled"})

@app.route("/api/schedules")
def get_schedules():
    return jsonify([{"id": j.id, "next_run": str(j.next_run_time)}
                    for j in scheduler.get_jobs()])

@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory("outputs", filename)

@app.route("/temp/<path:filename>")
def serve_temp(filename):
    return send_from_directory("temp", filename)

def _next_time(hour, minute):
    from datetime import timedelta
    now = datetime.now()
    t = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if t <= now: t += timedelta(days=1)
    return t

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("temp", exist_ok=True)
    app.run(host="0.0.0.0", port=7860, debug=False)
