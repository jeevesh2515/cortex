import pytest

from agent_os.health_check import check_payload


def payload():
    return {
        "schema_version": 1,
        "generated_at": "2026-09-13T00:00:00+00:00",
        "privacy": {
            "note_content_exposed": False,
            "paths_exposed": False,
            "credentials_exposed": False,
            "model_policy": "cloud_only",
        },
        "brain": {
            "vault_available": False,
            "note_count": 0,
            "folder_count": 0,
            "graphify_available": False,
        },
        "agents": {"scout": {"model": "cloud", "mode": "supervised"}},
        "trust": [],
    }


def test_health_check_accepts_sanitized_payload():
    result = check_payload(payload())
    assert result["status"] == "pass"
    assert result["sensitive_fields_detected"] == 0


@pytest.mark.parametrize("marker", ["/Users/private", "sk-secret", "prompt text", "vault_path"])
def test_health_check_rejects_sensitive_payload(marker):
    data = payload()
    data["agents"]["scout"]["model"] = marker
    with pytest.raises(ValueError):
        check_payload(data)
