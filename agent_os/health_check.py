#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

FORBIDDEN = (
    "/users/",
    "/home/",
    "sk-",
    "ghp_",
    "github_pat_",
    "bearer ",
    "password",
    "prompt",
    "secret",
    "vault_path",
    "graphify_artifacts",
)


def check_payload(payload: dict) -> dict:
    required = {"schema_version", "generated_at", "privacy", "brain", "agents", "trust"}
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"missing telemetry fields: {', '.join(missing)}")
    if payload["schema_version"] != 1:
        raise ValueError("unsupported telemetry schema version")
    privacy = payload["privacy"]
    expected = {
        "note_content_exposed": False,
        "paths_exposed": False,
        "credentials_exposed": False,
        "model_policy": "cloud_only",
    }
    if privacy != expected:
        raise ValueError("privacy contract mismatch")
    text = json.dumps(payload, separators=(",", ":")).lower()
    found = [item for item in FORBIDDEN if item in text]
    if found:
        raise ValueError(f"forbidden telemetry markers: {', '.join(found)}")
    for name, agent in payload["agents"].items():
        if not isinstance(name, str) or not isinstance(agent, dict):
            raise ValueError("invalid agent roster entry")
        if agent.get("mode") not in {"supervised", "disabled"}:
            raise ValueError(f"unsupported mode for agent {name}")
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "schema_version": payload["schema_version"],
        "agent_count": len(payload["agents"]),
        "trust_type_count": len(payload["trust"]),
        "sensitive_fields_detected": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8765/api/analytics")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        with urllib.request.urlopen(args.url, timeout=10) as response:
            if response.status != 200:
                raise ValueError(f"dashboard returned HTTP {response.status}")
            payload = json.load(response)
        result = check_payload(payload)
    except Exception as exc:  # noqa: BLE001 - CLI must return a useful health failure.
        result = {
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "status": "fail",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        print(json.dumps(result, indent=2), file=sys.stderr)
        if args.output:
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        return 1
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
