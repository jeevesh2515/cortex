#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT"
if [ "${1:-}" = "--response-json" ]; then exec python3 agent_os/loop.py "$@"; fi
RUN_ID=${CORTEX_OS_TASK_ID:-daily-cycle-$(date -u +%Y%m%dT%H%M%SZ)}
WORK=$(mktemp -d "${TMPDIR:-/tmp}/cortex-cloud-cycle.XXXXXX")
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/context"
git archive --format=tar HEAD | tar -x -C "$WORK/context"
rm -rf "$WORK/context/frontend/node_modules" "$WORK/context/graphify-out" "$WORK/context/.git"
mkdir -p "$WORK/context/agent_os" "$WORK/context/goals"
cp agent_os/agents.yaml "$WORK/context/agent_os/"
[ ! -f agent_os/README.md ] || cp agent_os/README.md "$WORK/context/agent_os/"
cp goals/README.md "$WORK/context/goals/"
SCOUT="$WORK/scout.json"; MANAGER="$WORK/manager.json"
SCOUT_PROMPT='You are the Cortex OS Scout. Use cloud inference only. Read-only. Do not edit, commit, push, deploy, call external services, or use local models. Inspect this clean repository snapshot only. Return exactly one JSON object and nothing else with keys attention_required, findings, suggested_task_type, confidence. Keep it concise.'
MANAGER_PROMPT='You are the Cortex OS Manager. Use cloud inference only. Do not edit files or execute work. Convert the Scout JSON below into exactly one JSON object and nothing else with keys objective, tasks, acceptance_tests, risk, status. tasks and acceptance_tests must be arrays. Keep the plan bounded and supervised. Scout JSON:'
valid_json() { [ -s "$1" ] && python3 -m json.tool "$1" >/dev/null 2>/dev/null; }
run_copilot() { _out=$1; _prompt=$2; if command -v copilot >/dev/null 2>&1; then (cd "$WORK/context" && timeout 60 copilot -p "$_prompt" --plan --silent --output-format text) >"$_out.raw" 2>"$_out.events" || true; elif command -v gh >/dev/null 2>&1; then (cd "$WORK/context" && timeout 60 gh copilot -- -p "$_prompt" --plan --silent --output-format text) >"$_out.raw" 2>"$_out.events" || true; else : >"$_out.raw"; : >"$_out.events"; fi; python3 agent_os/extract_json.py "$_out.raw" "$_out" || true; }
run_nvidia() { _out=$1; _prompt=$2; timeout 60 opencode run --pure --dir "$WORK/context" --model nvidia/openai/gpt-oss-20b --format json "$_prompt" >"$_out.raw" 2>"$_out.events" || true; python3 agent_os/extract_json.py "$_out.raw" "$_out" || true; }
run_cloud() { _out=$1; _prompt=$2; run_copilot "$_out" "$_prompt"; if valid_json "$_out"; then ACTIVE_MODEL=github-copilot-cli; return 0; fi; run_nvidia "$_out" "$_prompt"; if valid_json "$_out"; then ACTIVE_MODEL=nvidia/openai/gpt-oss-20b; return 0; fi; return 1; }
if ! run_cloud "$SCOUT" "$SCOUT_PROMPT"; then
  printf '%s\n' '{"http_status":200,"error":"No cloud Scout route produced valid JSON","stop_reason":"incomplete","model":"cloud-fallbacks"}' > "$WORK/fail.json"; CORTEX_OS_TASK_TYPE=scout CORTEX_OS_TASK_ID="$RUN_ID-scout" python3 agent_os/loop.py --response-json "$WORK/fail.json"; printf '%s\n' '{"http_status":200,"blocked":true,"error":"Manager blocked because Scout output was missing or invalid","stop_reason":"blocked","model":"cloud-fallbacks"}' > "$WORK/manager-blocked.json"; CORTEX_OS_TASK_TYPE=manager CORTEX_OS_TASK_ID="$RUN_ID-manager" python3 agent_os/loop.py --response-json "$WORK/manager-blocked.json"; exit 1
fi
SCOUT_JSON=$(cat "$SCOUT")
if ! run_cloud "$MANAGER" "$MANAGER_PROMPT $SCOUT_JSON"; then printf '%s\n' '{"http_status":200,"error":"No cloud Manager route produced valid JSON","stop_reason":"incomplete","model":"cloud-fallbacks"}' > "$WORK/fail.json"; CORTEX_OS_TASK_TYPE=manager CORTEX_OS_TASK_ID="$RUN_ID-manager" python3 agent_os/loop.py --response-json "$WORK/fail.json"; exit 1; fi
printf '%s\n' "{\"http_status\":200,\"response\":\"Scout and Manager completed\",\"stop_reason\":\"stop\",\"model\":\"$ACTIVE_MODEL\"}" > "$WORK/pass.json"
CORTEX_OS_TASK_TYPE=scout_manager_cycle CORTEX_OS_TASK_ID="$RUN_ID" python3 agent_os/loop.py --response-json "$WORK/pass.json"
printf '%s\n' "--- MODEL: $ACTIVE_MODEL ---"; printf '%s\n' '--- SCOUT ---'; cat "$SCOUT"; printf '%s\n' '--- MANAGER ---'; cat "$MANAGER"
