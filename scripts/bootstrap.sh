#!/usr/bin/env bash
# One-command setup for Cortex on macOS.
set -euo pipefail

BLUE='\033[0;34m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info() { echo -e "${BLUE}==>${NC} $*"; }
ok()   { echo -e "${GREEN} ok${NC} $*"; }
warn() { echo -e "${YELLOW}  !${NC} $*"; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

info "Checking prerequisites"

if ! command -v uv >/dev/null 2>&1; then
  warn "uv not found, installing"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
ok "uv $(uv --version | awk '{print $2}')"

info "Creating virtualenv"
uv venv --python 3.12
ok "$ROOT/.venv"

info "Installing Cortex"
uv pip install -e ".[all]"
ok "installed"

if command -v ollama >/dev/null 2>&1; then
  if ! curl -sf http://localhost:11434/api/version >/dev/null 2>&1; then
    warn "Ollama is installed but not running. Start it, then re-run this script."
  else
    info "Pulling models (this takes a few minutes)"
    ollama pull qwen3-embedding:0.6b
    ollama pull qwen3:4b
    ok "models ready"
  fi
else
  warn "Ollama not found. Install from https://ollama.com, then:"
  warn "  ollama pull qwen3-embedding:0.6b"
  warn "  ollama pull qwen3:4b"
  warn "Until then, use --offline to try the pipeline without a model."
fi

CONFIG_DIR="$HOME/.config/cortex"
CONFIG="$CONFIG_DIR/cortex.toml"
if [ ! -f "$CONFIG" ]; then
  info "Writing starter config"
  mkdir -p "$CONFIG_DIR"
  cp cortex.example.toml "$CONFIG"
  ok "$CONFIG"
  warn "Edit vault_path in that file before indexing."
else
  ok "config already exists at $CONFIG"
fi

echo
ok "Done. Next:"
echo "     1. Set vault_path in $CONFIG"
echo "     2. source .venv/bin/activate"
echo "     3. cortex index"
echo "     4. cortex ask \"what have I written about X?\""
