"""
Full test suite for youtube-agent.
Covers: database, Flask API, agent helper logic.
Heavy ML deps (torch/TTS/diffusers) are mocked out.
"""
import json
import os
import sys
import base64
import tempfile
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path

# ── Point at project root ─────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Patch heavy deps before any project import ────────────────────────────────
for mod in [
    "torch", "numpy", "soundfile", "anthropic",
    "TTS", "TTS.api", "audiocraft", "audiocraft.models",
    "diffusers", "transformers", "accelerate",
]:
    sys.modules.setdefault(mod, MagicMock())


# =============================================================================
# database.py
# =============================================================================

class TestJobDB(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        import database as _db_mod
        self._db_mod = _db_mod
        self._orig_path = _db_mod.DB_PATH
        _db_mod.DB_PATH = self.tmp.name
        from database import JobDB
        self.db = JobDB()

    def tearDown(self):
        self._db_mod.DB_PATH = self._orig_path
        os.unlink(self.tmp.name)

    # ── create / read ─────────────────────────────────────────────────────────

    def test_create_and_get_job(self):
        self.db.create_job("abc123", {"topic": "cats"})
        job = self.db.get_job("abc123")
        self.assertIsNotNone(job)
        self.assertEqual(job["id"], "abc123")
        self.assertEqual(job["status"], "pending")
        self.assertEqual(job["config"]["topic"], "cats")
        self.assertEqual(job["progress"], 0)

    def test_get_missing_job_returns_none(self):
        self.assertIsNone(self.db.get_job("does_not_exist"))

    def test_get_all_jobs_empty(self):
        self.assertEqual(self.db.get_all_jobs(), [])

    def test_get_all_jobs_multiple(self):
        self.db.create_job("j1", {"topic": "a"})
        self.db.create_job("j2", {"topic": "b"})
        jobs = self.db.get_all_jobs()
        self.assertEqual(len(jobs), 2)
        ids = {j["id"] for j in jobs}
        self.assertIn("j1", ids)
        self.assertIn("j2", ids)

    def test_get_all_jobs_capped_at_50(self):
        for i in range(55):
            self.db.create_job(f"job{i:03d}", {"topic": str(i)})
        jobs = self.db.get_all_jobs()
        self.assertLessEqual(len(jobs), 50)

    # ── update ────────────────────────────────────────────────────────────────

    def test_update_status(self):
        self.db.create_job("upd1", {"topic": "x"})
        self.db.update_job("upd1", status="running", progress=25)
        job = self.db.get_job("upd1")
        self.assertEqual(job["status"], "running")
        self.assertEqual(job["progress"], 25)

    def test_update_output_path(self):
        self.db.create_job("upd2", {"topic": "x"})
        self.db.update_job("upd2", output_path="/outputs/final.mp4")
        job = self.db.get_job("upd2")
        self.assertEqual(job["output_path"], "/outputs/final.mp4")

    # ── logs ──────────────────────────────────────────────────────────────────

    def test_add_log_appends(self):
        self.db.create_job("log1", {"topic": "x"})
        self.db.add_log("log1", "first message")
        self.db.add_log("log1", "second message")
        job = self.db.get_job("log1")
        self.assertEqual(len(job["logs"]), 2)
        self.assertEqual(job["logs"][0]["msg"], "first message")
        self.assertEqual(job["logs"][1]["msg"], "second message")

    def test_add_log_missing_job_no_crash(self):
        self.db.add_log("ghost", "should not crash")

    def test_log_has_time_key(self):
        self.db.create_job("log2", {"topic": "x"})
        self.db.add_log("log2", "hi")
        job = self.db.get_job("log2")
        self.assertIn("time", job["logs"][0])

    # ── photo approval ────────────────────────────────────────────────────────

    def test_set_and_get_pending_approval(self):
        self.db.create_job("pa1", {"topic": "x"})
        self.db.set_pending_approval("pa1", "pa1_s001", "/temp/img.jpg")
        data = self.db.get_pending_approval("pa1")
        self.assertIsNotNone(data)
        self.assertEqual(data["scene_id"], "pa1_s001")
        self.assertEqual(data["image_path"], "/temp/img.jpg")
        self.assertIsNone(data["decision"])

    def test_get_pending_approval_none_when_unset(self):
        self.db.create_job("pa2", {"topic": "x"})
        self.assertIsNone(self.db.get_pending_approval("pa2"))

    def test_set_approval_approved(self):
        self.db.create_job("pa3", {"topic": "x"})
        self.db.set_pending_approval("pa3", "pa3_s001", "/img.jpg")
        self.db.set_approval("pa3", "pa3_s001", "approved")
        data = self.db.get_pending_approval("pa3")
        self.assertEqual(data["decision"], "approved")

    def test_set_approval_rejected(self):
        self.db.create_job("pa4", {"topic": "x"})
        self.db.set_pending_approval("pa4", "pa4_s001", "/img.jpg")
        self.db.set_approval("pa4", "pa4_s001", "rejected")
        data = self.db.get_pending_approval("pa4")
        self.assertEqual(data["decision"], "rejected")

    def test_set_approval_wrong_scene_id_ignored(self):
        self.db.create_job("pa5", {"topic": "x"})
        self.db.set_pending_approval("pa5", "pa5_s001", "/img.jpg")
        self.db.set_approval("pa5", "wrong_scene", "approved")
        data = self.db.get_pending_approval("pa5")
        self.assertIsNone(data["decision"])

    def test_wait_for_approval_timeout_returns_approved(self):
        self.db.create_job("pa6", {"topic": "x"})
        self.db.set_pending_approval("pa6", "pa6_s001", "/img.jpg")
        result = self.db.wait_for_approval("pa6", "pa6_s001", timeout=0)
        self.assertEqual(result, "approved")

    def test_wait_for_approval_resolves_when_decision_set(self):
        import threading, time
        self.db.create_job("pa7", {"topic": "x"})
        self.db.set_pending_approval("pa7", "pa7_s001", "/img.jpg")

        def approve_after_delay():
            time.sleep(0.1)
            self.db.set_approval("pa7", "pa7_s001", "rejected")

        t = threading.Thread(target=approve_after_delay)
        t.start()
        result = self.db.wait_for_approval("pa7", "pa7_s001", timeout=5)
        t.join()
        self.assertEqual(result, "rejected")
        # pending_approval should be cleared after resolution
        self.assertIsNone(self.db.get_pending_approval("pa7"))


# =============================================================================
# app.py  (Flask API)
# =============================================================================

class TestFlaskApp(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()

        import database as _db_mod
        _db_mod.DB_PATH = self.tmp.name

        # Patch heavy agent deps
        patcher_agent = patch("agent.client", MagicMock())
        patcher_vgen  = patch("agent.VideoGenerator", MagicMock())
        patcher_igen  = patch("agent.ImageGenerator", MagicMock())
        patcher_agen  = patch("agent.AudioGenerator", MagicMock())
        patcher_mgen  = patch("agent.VideoMerger",    MagicMock())
        for p in (patcher_agent, patcher_vgen, patcher_igen, patcher_agen, patcher_mgen):
            p.start()
            self.addCleanup(p.stop)

        import app as _app_mod
        _app_mod.db = _app_mod.JobDB()
        _app_mod.app.config["TESTING"] = True
        self.client = _app_mod.app.test_client()
        self._app_mod = _app_mod

    def tearDown(self):
        os.unlink(self.tmp.name)

    def _post_json(self, url, data):
        return self.client.post(url, json=data,
                                content_type="application/json")

    # ── GET /api/jobs ─────────────────────────────────────────────────────────

    def test_list_jobs_empty(self):
        r = self.client.get("/api/jobs")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(json.loads(r.data), [])

    # ── POST /api/jobs ────────────────────────────────────────────────────────

    @patch("app._schedule_job")
    def test_create_job_returns_201(self, mock_sched):
        r = self._post_json("/api/jobs", {"topic": "test topic"})
        self.assertEqual(r.status_code, 201)
        body = json.loads(r.data)
        self.assertIn("job_id", body)

    @patch("app._schedule_job")
    def test_create_job_stores_config(self, mock_sched):
        r = self._post_json("/api/jobs", {
            "topic": "AI robots",
            "resolution": "1080p",
            "style": "documentary",
        })
        job_id = json.loads(r.data)["job_id"]
        job_r  = self.client.get(f"/api/jobs/{job_id}")
        job    = json.loads(job_r.data)
        self.assertEqual(job["config"]["topic"],      "AI robots")
        self.assertEqual(job["config"]["resolution"], "1080p")
        self.assertEqual(job["config"]["style"],      "documentary")

    @patch("app._schedule_job")
    def test_create_job_defaults(self, mock_sched):
        r = self._post_json("/api/jobs", {"topic": "default test"})
        job_id = json.loads(r.data)["job_id"]
        job    = json.loads(self.client.get(f"/api/jobs/{job_id}").data)
        cfg    = job["config"]
        self.assertEqual(cfg["resolution"],  "720p")
        self.assertEqual(cfg["video_model"], "wan22")
        self.assertEqual(cfg["fps"],         16)
        self.assertEqual(cfg["steps"],       15)

    @patch("app._schedule_job")
    def test_create_job_empty_body(self, mock_sched):
        r = self._post_json("/api/jobs", {})
        self.assertEqual(r.status_code, 201)

    # ── GET /api/jobs/:id ─────────────────────────────────────────────────────

    def test_get_missing_job_returns_404(self):
        r = self.client.get("/api/jobs/nonexistent")
        self.assertEqual(r.status_code, 404)

    @patch("app._schedule_job")
    def test_get_existing_job(self, mock_sched):
        r    = self._post_json("/api/jobs", {"topic": "x"})
        jid  = json.loads(r.data)["job_id"]
        resp = self.client.get(f"/api/jobs/{jid}")
        self.assertEqual(resp.status_code, 200)
        job  = json.loads(resp.data)
        self.assertEqual(job["id"],     jid)
        self.assertEqual(job["status"], "pending")

    # ── POST /api/jobs/:id/cancel ─────────────────────────────────────────────

    @patch("app._schedule_job")
    def test_cancel_job(self, mock_sched):
        r   = self._post_json("/api/jobs", {"topic": "cancel me"})
        jid = json.loads(r.data)["job_id"]
        rc  = self.client.post(f"/api/jobs/{jid}/cancel")
        self.assertEqual(rc.status_code, 200)
        job = json.loads(self.client.get(f"/api/jobs/{jid}").data)
        self.assertEqual(job["status"], "cancelled")

    # ── POST /api/jobs/:id/approve ────────────────────────────────────────────

    @patch("app._schedule_job")
    def test_approve_photo(self, mock_sched):
        r    = self._post_json("/api/jobs", {"topic": "approve"})
        jid  = json.loads(r.data)["job_id"]
        self._app_mod.db.set_pending_approval(jid, f"{jid}_s001", "/tmp/img.jpg")
        ra   = self._post_json(f"/api/jobs/{jid}/approve",
                               {"scene_id": f"{jid}_s001", "decision": "approved"})
        self.assertEqual(ra.status_code, 200)
        data = self._app_mod.db.get_pending_approval(jid)
        self.assertEqual(data["decision"], "approved")

    # ── POST /api/generate ────────────────────────────────────────────────────

    @patch("app._schedule_job")
    def test_generate_multi_single_topic(self, mock_sched):
        r    = self._post_json("/api/generate", {"topics": "space", "count": 2})
        self.assertEqual(r.status_code, 201)
        body = json.loads(r.data)
        self.assertEqual(len(body["job_ids"]), 2)

    @patch("app._schedule_job")
    def test_generate_multi_two_topics(self, mock_sched):
        r    = self._post_json("/api/generate", {"topics": "cats\ndogs", "count": 1})
        body = json.loads(r.data)
        self.assertEqual(len(body["job_ids"]), 2)

    @patch("app._schedule_job")
    def test_list_jobs_after_create(self, mock_sched):
        self._post_json("/api/jobs", {"topic": "a"})
        self._post_json("/api/jobs", {"topic": "b"})
        r    = self.client.get("/api/jobs")
        jobs = json.loads(r.data)
        self.assertEqual(len(jobs), 2)

    # ── Alias routes ──────────────────────────────────────────────────────────

    @patch("app._schedule_job")
    def test_cancel_alias_route(self, mock_sched):
        r   = self._post_json("/api/jobs", {"topic": "alias cancel"})
        jid = json.loads(r.data)["job_id"]
        rc  = self.client.post(f"/api/job/{jid}/cancel")
        self.assertEqual(rc.status_code, 200)

    @patch("app._schedule_job")
    def test_approve_alias_route(self, mock_sched):
        r    = self._post_json("/api/jobs", {"topic": "alias approve"})
        jid  = json.loads(r.data)["job_id"]
        self._app_mod.db.set_pending_approval(jid, f"{jid}_s001", "/img.jpg")
        ra   = self._post_json(f"/api/job/{jid}/approve_photo",
                               {"scene_id": f"{jid}_s001"})
        self.assertEqual(ra.status_code, 200)

    @patch("app._schedule_job")
    def test_reject_alias_route(self, mock_sched):
        r    = self._post_json("/api/jobs", {"topic": "alias reject"})
        jid  = json.loads(r.data)["job_id"]
        self._app_mod.db.set_pending_approval(jid, f"{jid}_s001", "/img.jpg")
        rr   = self._post_json(f"/api/job/{jid}/reject_photo",
                               {"scene_id": f"{jid}_s001"})
        self.assertEqual(rr.status_code, 200)

    # ── Upload voice ──────────────────────────────────────────────────────────

    def test_upload_voice_no_file_returns_400(self):
        r = self.client.post("/api/upload_voice")
        self.assertEqual(r.status_code, 400)

    # ── Static pages ─────────────────────────────────────────────────────────

    def test_index_page_200(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)


# =============================================================================
# agent.py  (helper logic — no ML inference)
# =============================================================================

class TestAgentHelpers(unittest.TestCase):
    def setUp(self):
        with patch("agent.VideoGenerator", MagicMock()), \
             patch("agent.ImageGenerator", MagicMock()), \
             patch("agent.AudioGenerator", MagicMock()), \
             patch("agent.VideoMerger",    MagicMock()), \
             patch("agent.client",         MagicMock()), \
             patch("agent.db",             MagicMock()):
            from agent import VideoAgent
            self.agent = VideoAgent()

    # ── _resolve_duration ─────────────────────────────────────────────────────

    def test_resolve_duration_auto_uses_scene_value(self):
        scene = {"duration_seconds": 15}
        self.assertEqual(self.agent._resolve_duration(scene, "auto"), 15)

    def test_resolve_duration_auto_default_10(self):
        self.assertEqual(self.agent._resolve_duration({}, "auto"), 10)

    def test_resolve_duration_fixed_ignores_scene(self):
        scene = {"duration_seconds": 99}
        self.assertEqual(self.agent._resolve_duration(scene, "5"),  5)
        self.assertEqual(self.agent._resolve_duration(scene, "20"), 20)

    # ── _dominant_mood ────────────────────────────────────────────────────────

    def test_dominant_mood_majority(self):
        scenes = [
            {"music_mood": "epic"},
            {"music_mood": "epic"},
            {"music_mood": "calm"},
        ]
        self.assertEqual(self.agent._dominant_mood(scenes), "epic")

    def test_dominant_mood_single(self):
        scenes = [{"music_mood": "dramatic"}]
        self.assertEqual(self.agent._dominant_mood(scenes), "dramatic")

    def test_dominant_mood_default_cinematic(self):
        scenes = [{}]
        self.assertEqual(self.agent._dominant_mood(scenes), "cinematic")

    # ── _build_narration ──────────────────────────────────────────────────────

    def test_build_narration_joins_parts(self):
        plan   = {"narration_intro": "Intro.", "narration_outro": "Outro."}
        scenes = [{"narration": "Scene 1."}, {"narration": "Scene 2."}]
        result = self.agent._build_narration(plan, scenes)
        self.assertIn("Intro.",   result)
        self.assertIn("Scene 1.", result)
        self.assertIn("Scene 2.", result)
        self.assertIn("Outro.",   result)

    def test_build_narration_skips_empty(self):
        plan   = {"narration_intro": "", "narration_outro": ""}
        scenes = [{"narration": "Only scene."}]
        result = self.agent._build_narration(plan, scenes)
        self.assertEqual(result.strip(), "Only scene.")

    def test_build_narration_no_scenes(self):
        plan   = {"narration_intro": "Solo.", "narration_outro": ""}
        result = self.agent._build_narration(plan, [])
        self.assertEqual(result.strip(), "Solo.")

    # ── _manual_plan ─────────────────────────────────────────────────────────

    def test_manual_plan_single_paragraph(self):
        plan = self.agent._manual_plan("Robots dance in space.", "Robot Dance", "5")
        self.assertEqual(plan["title"], "Robot Dance")
        self.assertEqual(len(plan["scenes"]), 1)
        self.assertEqual(plan["scenes"][0]["video_prompt"], "Robots dance in space.")
        self.assertEqual(plan["scenes"][0]["duration_seconds"], 5)

    def test_manual_plan_multiple_paragraphs(self):
        text = "Scene A\n\nScene B\n\nScene C"
        plan = self.agent._manual_plan(text, "Multi", "10")
        self.assertEqual(len(plan["scenes"]), 3)

    def test_manual_plan_auto_clip_duration(self):
        plan = self.agent._manual_plan("One scene.", "Topic", "auto")
        self.assertEqual(plan["scenes"][0]["duration_seconds"], 10)

    def test_manual_plan_first_scene_is_key(self):
        plan = self.agent._manual_plan("A\nB\nC", "T", "5")
        self.assertTrue(plan["scenes"][0]["is_key_scene"])
        self.assertFalse(plan["scenes"][1]["is_key_scene"])

    def test_manual_plan_scene_ids_sequential(self):
        plan = self.agent._manual_plan("A\nB\nC", "T", "5")
        ids  = [s["id"] for s in plan["scenes"]]
        self.assertEqual(ids, [1, 2, 3])

    def test_manual_plan_has_intro_outro(self):
        plan = self.agent._manual_plan("X", "T", "5")
        self.assertIn("narration_intro", plan)
        self.assertIn("narration_outro", plan)

    # ── _save_uploads ─────────────────────────────────────────────────────────

    def test_save_uploads_empty(self):
        paths = self.agent._save_uploads([], "job99")
        self.assertEqual(paths, [])

    def test_save_uploads_valid_base64(self):
        raw   = b"fake image data"
        b64   = "data:image/jpeg;base64," + base64.b64encode(raw).decode()
        paths = self.agent._save_uploads([b64], "job99")
        self.assertEqual(len(paths), 1)
        saved = Path(paths[0])
        try:
            self.assertTrue(saved.exists())
            self.assertEqual(saved.read_bytes(), raw)
        finally:
            saved.unlink(missing_ok=True)

    def test_save_uploads_invalid_b64_no_crash(self):
        paths = self.agent._save_uploads(["not_valid_base64!!!"], "job98")
        self.assertEqual(paths, [])


# =============================================================================
# app.py — schedule helper
# =============================================================================

class TestScheduleHelper(unittest.TestCase):
    def setUp(self):
        import database as _db_mod
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        _db_mod.DB_PATH = self.tmp.name

        for mod in ("agent.VideoGenerator", "agent.ImageGenerator",
                    "agent.AudioGenerator", "agent.VideoMerger", "agent.client"):
            patch(mod, MagicMock()).start()

        import app as _app
        _app.db = _app.JobDB()
        self._app = _app

    def tearDown(self):
        os.unlink(self.tmp.name)

    @patch("app.threading.Thread")
    def test_schedule_now_starts_thread(self, mock_thread):
        mock_instance = MagicMock()
        mock_thread.return_value = mock_instance
        self._app._schedule_job("job1", {"type": "now"})
        mock_thread.assert_called_once()
        mock_instance.start.assert_called_once()

    def test_schedule_once_adds_to_scheduler(self):
        with patch.object(self._app.scheduler, "add_job") as mock_add:
            self._app._schedule_job("job2", {
                "type": "once",
                "run_date": "2099-01-01T09:00:00"
            })
            mock_add.assert_called_once()
            call_kwargs = mock_add.call_args
            self.assertEqual(call_kwargs[1]["id"], "job2")

    def test_schedule_daily_adds_cron_job(self):
        with patch.object(self._app.scheduler, "add_job") as mock_add:
            self._app._schedule_job("job3", {"type": "daily", "time": "07:30"})
            mock_add.assert_called_once()
            call_args = mock_add.call_args
            self.assertEqual(call_args[1]["hour"],   7)
            self.assertEqual(call_args[1]["minute"], 30)


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
