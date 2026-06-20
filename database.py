"""
SQLite database for job tracking + photo approvals
"""
import sqlite3, json
from datetime import datetime

DB_PATH = "agent_jobs.db"

class JobDB:
    def __init__(self):
        self._init_db()

    def _conn(self):
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    config TEXT,
                    status TEXT DEFAULT 'pending',
                    progress INTEGER DEFAULT 0,
                    progress_text TEXT DEFAULT '',
                    output_path TEXT DEFAULT '',
                    created_at TEXT,
                    updated_at TEXT,
                    logs TEXT DEFAULT '[]',
                    pending_approval TEXT DEFAULT NULL
                )
            """)

    def create_job(self, job_id, config: dict):
        now = datetime.now().isoformat()
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO jobs (id, config, status, progress, created_at, updated_at)
                VALUES (?, ?, 'pending', 0, ?, ?)
            """, (job_id, json.dumps(config, ensure_ascii=False), now, now))

    def update_job(self, job_id, **kwargs):
        kwargs["updated_at"] = datetime.now().isoformat()
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        vals = list(kwargs.values()) + [job_id]
        with self._conn() as conn:
            conn.execute(f"UPDATE jobs SET {sets} WHERE id = ?", vals)

    def add_log(self, job_id, message):
        with self._conn() as conn:
            row = conn.execute("SELECT logs FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row:
                logs = json.loads(row["logs"])
                logs.append({"time": datetime.now().strftime("%H:%M:%S"), "msg": message})
                conn.execute("UPDATE jobs SET logs = ?, updated_at = ? WHERE id = ?",
                    (json.dumps(logs), datetime.now().isoformat(), job_id))

    # ── Photo approval ────────────────────────────────────────────────────────

    def set_pending_approval(self, job_id, scene_id, image_path):
        """Pause job waiting for user to approve/reject a generated photo"""
        data = json.dumps({"scene_id": scene_id, "image_path": image_path, "decision": None})
        self.update_job(job_id, pending_approval=data)

    def get_pending_approval(self, job_id):
        with self._conn() as conn:
            row = conn.execute("SELECT pending_approval FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row and row["pending_approval"]:
                return json.loads(row["pending_approval"])
        return None

    def set_approval(self, job_id, scene_id, decision):
        """Set approved/rejected decision"""
        with self._conn() as conn:
            row = conn.execute("SELECT pending_approval FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row and row["pending_approval"]:
                data = json.loads(row["pending_approval"])
                if data.get("scene_id") == scene_id:
                    data["decision"] = decision
                    conn.execute("UPDATE jobs SET pending_approval = ? WHERE id = ?",
                        (json.dumps(data), job_id))

    def wait_for_approval(self, job_id, scene_id, timeout=300):
        """Block until user approves or rejects (called from agent thread)"""
        import time
        start = time.time()
        while time.time() - start < timeout:
            data = self.get_pending_approval(job_id)
            if data and data.get("scene_id") == scene_id and data.get("decision"):
                self.update_job(job_id, pending_approval=None)
                return data["decision"]
            time.sleep(1)
        self.add_log(job_id, "⏰ Таймаут підтвердження — авто-схвалено")
        return "approved"

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_job(self, job_id):
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row:
                d = dict(row)
                d["logs"] = json.loads(d["logs"])
                d["config"] = json.loads(d["config"]) if d.get("config") else {}
                if d.get("pending_approval"):
                    d["pending_approval"] = json.loads(d["pending_approval"])
                return d
        return None

    def get_all_jobs(self):
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT 50").fetchall()
            result = []
            for row in rows:
                d = dict(row)
                d["logs"] = json.loads(d["logs"])
                d["config"] = json.loads(d["config"]) if d.get("config") else {}
                if d.get("pending_approval"):
                    d["pending_approval"] = json.loads(d["pending_approval"])
                result.append(d)
            return result
