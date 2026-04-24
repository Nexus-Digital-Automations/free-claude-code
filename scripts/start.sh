#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PID_FILE="$ROOT/logs/server.pid"
LOG_FILE="$ROOT/logs/server.log"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8082}"
HEALTH_URL="http://$HOST:$PORT/health"

mkdir -p "$ROOT/logs"

# Already running?
if [[ -f "$PID_FILE" ]]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Server is already running (PID $OLD_PID) on $HOST:$PORT"
        exit 0
    else
        echo "Stale PID file found, cleaning up..."
        rm -f "$PID_FILE"
    fi
fi

# Clear anything on the port before starting
if lsof -ti:"$PORT" &>/dev/null; then
    echo "Port $PORT is in use — clearing it..."
    lsof -ti:"$PORT" | xargs kill -9 2>/dev/null || true
    sleep 0.5
fi

echo "Starting free-claude-code proxy on $HOST:$PORT..."

cd "$ROOT"
nohup uv run uvicorn server:app \
    --host "$HOST" \
    --port "$PORT" \
    >> "$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"

# Wait for health check (up to 15 seconds)
echo -n "Waiting for server to be ready"
for i in $(seq 1 30); do
    if curl -sf "$HEALTH_URL" &>/dev/null; then
        echo ""
        echo "Server is up (PID $SERVER_PID)"
        echo ""
        echo "  Base URL : http://$HOST:$PORT"
        echo "  Logs     : $LOG_FILE"
        echo "  Stop with: $SCRIPT_DIR/stop.sh"
        echo ""
        echo "To use with Claude Code:"
        echo "  export ANTHROPIC_BASE_URL=http://$HOST:$PORT"
        echo "  export ANTHROPIC_API_KEY=dummy"
        exit 0
    fi
    # Check the process didn't die immediately
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo ""
        echo "Error: server process died. Check logs:"
        tail -20 "$LOG_FILE"
        rm -f "$PID_FILE"
        exit 1
    fi
    echo -n "."
    sleep 0.5
done

echo ""
echo "Error: server didn't become healthy within 15 seconds. Check logs:"
tail -20 "$LOG_FILE"
exit 1
