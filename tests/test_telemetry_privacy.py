import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_os.dashboard import analytics


def test_public_telemetry_is_aggregate_only(monkeypatch):
    payload = analytics()
    text = json.dumps(payload)
    assert payload["schema_version"] == 1
    assert payload["privacy"] == {
        "note_content_exposed": False,
        "paths_exposed": False,
        "credentials_exposed": False,
        "model_policy": "cloud_only",
    }
    for forbidden in ("/Users/", "/home/", "sk-", "ghp_", "github_pat_", "Bearer ", "password", "prompt", "secret"):
        assert forbidden.lower() not in text.lower()
    assert "folders" not in payload["brain"]
    assert "graphify_artifacts" not in payload["brain"]


def test_unknown_task_types_are_redacted(monkeypatch, tmp_path):
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"task_types": {"/Users/private/Note.md": {"completed": 1, "success_rate": 1}}}))
    monkeypatch.setattr("agent_os.dashboard.STATE", state)
    payload = analytics()
    assert payload["trust"][0]["type"] == "other"
    assert "/Users/" not in json.dumps(payload)
