#!/usr/bin/env python3
from __future__ import annotations
import json, os, re
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VAULT = Path(os.environ["CORTEX_VAULT"]).expanduser() if os.environ.get("CORTEX_VAULT") else Path("__cortex_vault_unconfigured__")
GRAPH = VAULT / "graphify-out"
STATE = ROOT / "agent_os/state.json"
MANIFEST = GRAPH / "manifest.json"


def read_json(path: Path, fallback):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return fallback


def analytics() -> dict:
    manifest = read_json(MANIFEST, {})
    state = read_json(STATE, {"task_types": {}})
    note_paths = []
    if VAULT.exists():
        note_paths = [str(p.relative_to(VAULT)) for p in VAULT.rglob("*.md") if ".obsidian" not in p.parts and "graphify-out" not in p.parts and ".trash" not in p.parts]
    if not note_paths:
        note_paths = [p for p in manifest if p.endswith(".md") and not p.startswith(".obsidian/")]
    folders: dict[str, int] = {}
    for name in note_paths:
        folder = name.split("/", 1)[0] if "/" in name else "root"
        folders[folder] = folders.get(folder, 0) + 1
    tasks = []
    allowed_types = {"scout", "manager", "scout_cloud_probe", "scout_manager_cycle", "health_check", "model_probe"}
    for kind, data in state.get("task_types", {}).items():
        tasks.append({"type": kind if kind in allowed_types else "other", "completed": max(0, int(data.get("completed", 0))), "success_rate": min(1, max(0, float(data.get("success_rate", 0)))), "privilege": data.get("privilege", "supervised") if data.get("privilege", "supervised") in {"supervised", "autonomous", "supervised_revoked"} else "supervised", "runs": min(100000, max(0, len(data.get("runs", []))))})
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "privacy": {"note_content_exposed": False, "paths_exposed": False, "credentials_exposed": False, "model_policy": "cloud_only"},
        "brain": {"vault_available": VAULT.exists(), "note_count": len(note_paths), "folder_count": len(folders), "graphify_available": GRAPH.exists()},
        "agents": {"scout": {"model": "github-copilot-cli", "fallback": "nvidia/openai/gpt-oss-20b", "mode": "supervised"}, "manager": {"model": "github-copilot-cli", "fallback": "nvidia/openai/gpt-oss-20b", "mode": "supervised"}, "builder": {"model": "unavailable_until_explicitly_verified", "mode": "disabled"}, "inspector": {"model": "unavailable_until_explicitly_verified", "mode": "disabled"}},
        "trust": tasks,
    }


HTML = """<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>Cortex OS — Command Center</title><style>
:root{color-scheme:dark;--bg:#090b12;--panel:#111522;--muted:#8d98ae;--text:#eef2ff;--cyan:#65e6d0;--violet:#a78bfa;--line:#242b3c}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 80% 0,#20204a 0,#090b12 42%);color:var(--text);font:14px ui-sans-serif,system-ui,-apple-system,sans-serif}main{max-width:1280px;margin:auto;padding:32px 24px}.eyebrow{color:var(--cyan);letter-spacing:.14em;text-transform:uppercase;font-size:11px;font-weight:700}.header{display:flex;justify-content:space-between;align-items:end;gap:24px;margin-bottom:28px}.header h1{font-size:42px;letter-spacing:-.05em;margin:8px 0}.header p{color:var(--muted);margin:0}.pill{border:1px solid #31534f;background:#102521;color:var(--cyan);border-radius:999px;padding:8px 12px}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.card{background:linear-gradient(145deg,#151a2a,#0f131e);border:1px solid var(--line);border-radius:18px;padding:18px;box-shadow:0 18px 50px #0003}.metric{font-size:32px;font-weight:750;margin-top:10px}.label{color:var(--muted);font-size:12px}.section{margin-top:18px}.section h2{font-size:16px;margin:0 0 12px}.wide{grid-column:span 2}.agent{display:flex;align-items:center;justify-content:space-between;padding:14px 0;border-bottom:1px solid var(--line)}.agent:last-child{border:0}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--cyan);margin-right:8px}.disabled .dot{background:#687086}.bar{height:7px;background:#252c3d;border-radius:99px;overflow:hidden;margin-top:10px}.bar i{display:block;height:100%;background:linear-gradient(90deg,var(--violet),var(--cyan));border-radius:99px}.small{font-size:12px;color:var(--muted)}.folder{display:flex;justify-content:space-between;padding:10px 0;border-bottom:1px solid var(--line)}@media(max-width:850px){.grid{grid-template-columns:repeat(2,1fr)}.wide{grid-column:span 2}}@media(max-width:560px){main{padding:20px 14px}.grid{grid-template-columns:1fr}.wide{grid-column:span 1}.header{display:block}.header h1{font-size:34px}.pill{display:inline-block;margin-top:16px}}
</style></head><body><main><div class=header><div><div class=eyebrow>Cortex OS · live command center</div><h1>Brain telemetry, agent control.</h1><p>Aggregate-only view of your Obsidian brain, Graphify map, and supervised cloud agents.</p></div><div class=pill id=updated>Connecting…</div></div><div class=grid><div class=card><div class=label>Knowledge notes</div><div class=metric id=notes>—</div><div class=small>Obsidian manifest</div></div><div class=card><div class=label>Graphify artifacts</div><div class=metric id=artifacts>—</div><div class=small>local index outputs</div></div><div class=card><div class=label>Active cloud agents</div><div class=metric id=active>—</div><div class=small>supervised only</div></div><div class=card><div class=label>Successful cycles</div><div class=metric id=cycles>—</div><div class=small>report-card history</div></div><div class='card wide'><h2>Agent roster</h2><div id=agents></div></div><div class='card wide'><h2>Trust report</h2><div id=trust></div></div><div class='card wide'><h2>Brain shape</h2><div id=folders></div></div><div class='card wide'><h2>Privacy boundary</h2><p class=small id=privacy></p></div></div></main><script>
async function load(){const d=await fetch('/api/analytics',{cache:'no-store'}).then(r=>r.json());document.querySelector('#updated').textContent='Updated '+new Date(d.generated_at).toLocaleTimeString();notes.textContent=d.brain.note_count;artifacts.textContent=d.brain.graphify_available?'available':'not detected';active.textContent=Object.values(d.agents).filter(x=>x.mode==='supervised').length;cycles.textContent=(d.trust.find(x=>x.type==='scout_manager_cycle')||{}).completed||0;agents.innerHTML=Object.entries(d.agents).map(([n,a])=>`<div class="agent ${a.mode==='disabled'?'disabled':''}"><span><i class=dot></i><b>${n}</b><div class=small>${a.model}</div></span><span class=small>${a.mode}</span></div>`).join('');trust.innerHTML=d.trust.map(t=>`<div class=agent><span><b>${t.type}</b><div class=small>${t.runs} recorded runs · ${t.privilege}</div></span><span>${Math.round(t.success_rate*100)}%</span></div><div class=bar><i style="width:${Math.round(t.success_rate*100)}%"></i></div>`).join('')||'<p class=small>No report-card events yet.</p>';folders.innerHTML=`<div class=folder><span>Vault availability</span><b>${d.brain.vault_available?'available':'not detected'}</b></div><div class=folder><span>Aggregate folders</span><b>${d.brain.folder_count}</b></div>`;privacy.textContent=`Only counts, booleans, model identifiers, task categories, grades, and timestamps are exposed. Note text, folder names, Graphify filenames, absolute paths, prompts, agent output, and credentials are excluded.`}load();setInterval(load,15000);
</script></body></html>"""

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/analytics":
            body = json.dumps(analytics()).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
        body = HTML.encode(); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def log_message(self, *_): pass

if __name__ == "__main__":
    port = int(os.environ.get("CORTEX_DASHBOARD_PORT", "8765"))
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
