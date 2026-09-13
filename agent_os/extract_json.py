#!/usr/bin/env python3
from __future__ import annotations
import json, re, sys
from pathlib import Path

def objects(text: str):
    decoder = json.JSONDecoder()
    for match in re.finditer(r'\{', text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            yield value

def main() -> int:
    if len(sys.argv) != 3:
        return 2
    raw = Path(sys.argv[1]).read_text(errors="replace")
    candidates = []
    for line in raw.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") in {"agent_message", "assistant_message"}:
                candidates.append(str(item.get("text", "")))
            part = event.get("part")
            if isinstance(part, dict) and part.get("type") == "text":
                candidates.append(str(part.get("text", "")))
            if event.get("type") in {"text", "agent_message"}:
                candidates.append(str(event.get("text", "")))
        candidates.append(line)
    candidates.append(raw)
    for candidate in reversed(candidates):
        for value in objects(candidate):
            if any(k in value for k in ("attention_required", "objective", "tasks", "findings")):
                Path(sys.argv[2]).write_text(json.dumps(value, indent=2) + "\n")
                return 0
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
