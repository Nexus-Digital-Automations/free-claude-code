#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PID_FILE="$ROOT/logs/server.pid"
PORT="${PORT:-8082}"

stop_pid() {
    local pid=$1
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "Process $pid is not running."
        return 0
    fi
    echo "Stopping server (PID $pid)..."
    kill "$pid"
    for i in $(seq 1 20); do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "Server stopped."
            return 0
        fi
        sleep 0.5
    done
    echo "Server didn't stop within 10 seconds — force-killing..."
    kill -9 "$pid" 2>/dev/null || true
    echo "Server force-killed."
}

if [[ -f "$PID_FILE" ]]; then
    PID=$(cat "$PID_FILE")
    stop_pid "$PID"
    rm -f "$PID_FILE"
elif lsof -ti:"$PORT" &>/dev/null; then
    PID=$(lsof -ti:"$PORT")
    echo "No PID file found; stopping process on port $PORT (PID $PID)..."
    stop_pid "$PID"
else
    echo "Server is not running."
fi
