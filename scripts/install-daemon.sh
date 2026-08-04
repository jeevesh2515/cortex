#!/usr/bin/env bash
# Install Cortex as a macOS LaunchDaemon.
#
# A Daemon, not an Agent: since macOS 26 Tahoe, Homebrew Python running as a
# LaunchAgent is subject to TCC checks and fails local network calls with
# errno 65. A LaunchDaemon with a UserName key avoids this.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="com.jeevesh.cortex"
PLIST="/Library/LaunchDaemons/${LABEL}.plist"
LOG_DIR="/usr/local/var/log/cortex"
PYTHON="$ROOT/.venv/bin/python"

if [ ! -x "$PYTHON" ]; then
  echo "Virtualenv not found at $PYTHON. Run ./scripts/bootstrap.sh first." >&2
  exit 1
fi

echo "==> Installing $LABEL (requires sudo)"
sudo mkdir -p "$LOG_DIR"
sudo chown "$(whoami)" "$LOG_DIR"

sudo tee "$PLIST" >/dev/null <<PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${LABEL}</string>

    <key>ProgramArguments</key>
    <array>
        <string>${PYTHON}</string>
        <string>-m</string>
        <string>cortex.cli</string>
        <string>watch</string>
        <string>--interval</string>
        <string>60</string>
    </array>

    <!-- Run as the invoking user, not root: the daemon needs access to the
         user's vault and Ollama, and must not create root-owned index files. -->
    <key>UserName</key>
    <string>$(whoami)</string>

    <key>WorkingDirectory</key>
    <string>${ROOT}</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>HOME</key>
        <string>${HOME}</string>
        <key>PATH</key>
        <string>/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin</string>
    </dict>

    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>

    <!-- Background priority: indexing must never contend with the user's work.
         The thermal governor handles heat; this handles CPU scheduling. -->
    <key>ProcessType</key>
    <string>Background</string>
    <key>Nice</key>
    <integer>5</integer>
    <key>LowPriorityIO</key>
    <true/>

    <key>ThrottleInterval</key>
    <integer>30</integer>

    <key>StandardOutPath</key>
    <string>${LOG_DIR}/cortex.log</string>
    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/cortex.err.log</string>
</dict>
</plist>
PLIST_EOF

sudo chown root:wheel "$PLIST"
sudo chmod 644 "$PLIST"

sudo launchctl bootout "system/${LABEL}" 2>/dev/null || true
sudo launchctl bootstrap system "$PLIST"
sudo launchctl enable "system/${LABEL}"

# launchd does not rotate logs; newsyslog does.
sudo tee "/etc/newsyslog.d/${LABEL}.conf" >/dev/null <<NEWSYSLOG
# logfilename                        [owner:group]  mode count size when  flags
${LOG_DIR}/cortex.log                $(whoami):staff 644  5     5120 *     J
${LOG_DIR}/cortex.err.log            $(whoami):staff 644  5     5120 *     J
NEWSYSLOG

echo "==> Installed."
echo "    status:  sudo launchctl print system/${LABEL}"
echo "    logs:    tail -f ${LOG_DIR}/cortex.log"
echo "    remove:  sudo launchctl bootout system/${LABEL} && sudo rm ${PLIST}"
