#!/usr/bin/env bash

# HeatWise: Unified Demo Launcher
# Starts both the static presentation landing page (port 8000)
# and the dynamic Streamlit backend application (port 8501).

set -e

# 1. Determine project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 2. Check Python availability
if command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON="python"
else
    echo "Error: Python is not installed or not in PATH."
    exit 1
fi

# 3. Check Streamlit installation
if ! "$PYTHON" -m streamlit --version >/dev/null 2>&1; then
    echo "Error: Streamlit is not installed for $PYTHON."
    echo "Please install dependencies with: $PYTHON -m pip install -r requirements.txt"
    exit 1
fi

# 4. Port Check (8000 and 8501)
check_port() {
    local port=$1
    if "$PYTHON" -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); res = s.connect_ex(('127.0.0.1', $port)); s.close(); exit(0 if res == 0 else 1)" 2>/dev/null; then
        echo "Port $port is already in use."
        if command -v lsof >/dev/null 2>&1; then
            local pids
            pids=$(lsof -t -i :"$port" 2>/dev/null | tr '\n' ' ')
            if [ -n "$pids" ]; then
                echo "Process occupying port $port: PID $pids"
                echo "To stop it, run: kill $pids"
            fi
        fi
        return 1
    fi
    return 0
}

PORT_ERROR=0
if ! check_port 8000; then
    PORT_ERROR=1
fi
if ! check_port 8501; then
    PORT_ERROR=1
fi

if [ $PORT_ERROR -ne 0 ]; then
    echo ""
    echo "Please stop the existing process(es) before starting HeatWise."
    exit 1
fi

# 5. Banner & URLs Output
echo "Landing Page:"
echo "http://localhost:8000"
echo "HeatWise Application:"
echo "http://localhost:8501"
echo "Starting services..."
echo "Landing page → 8000"
echo "Streamlit → 8501"
echo "Open:"
echo "http://localhost:8000"
echo "=================================================="
echo "Press Ctrl+C to stop all services."

# 6. Process Management & Cleanup
HTTP_PID=""
STREAMLIT_PID=""

cleanup() {
    trap - SIGINT SIGTERM EXIT
    echo ""
    echo "Shutting down HeatWise services..."
    if [ -n "$HTTP_PID" ] && kill -0 "$HTTP_PID" 2>/dev/null; then
        kill -TERM "$HTTP_PID" 2>/dev/null || true
    fi
    if [ -n "$STREAMLIT_PID" ] && kill -0 "$STREAMLIT_PID" 2>/dev/null; then
        kill -TERM "$STREAMLIT_PID" 2>/dev/null || true
    fi

    # Wait up to 2 seconds for graceful shutdown
    local waited=0
    while [ $waited -lt 20 ]; do
        local running=0
        if [ -n "$HTTP_PID" ] && kill -0 "$HTTP_PID" 2>/dev/null; then
            running=1
        fi
        if [ -n "$STREAMLIT_PID" ] && kill -0 "$STREAMLIT_PID" 2>/dev/null; then
            running=1
        fi
        [ $running -eq 0 ] && break
        sleep 0.1
        waited=$((waited + 1))
    done

    # Force kill if still lingering
    if [ -n "$HTTP_PID" ] && kill -0 "$HTTP_PID" 2>/dev/null; then
        kill -9 "$HTTP_PID" 2>/dev/null || true
    fi
    if [ -n "$STREAMLIT_PID" ] && kill -0 "$STREAMLIT_PID" 2>/dev/null; then
        kill -9 "$STREAMLIT_PID" 2>/dev/null || true
    fi

    echo "Services stopped cleanly."
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# Start landing page
"$PYTHON" -m http.server 8000 >/dev/null 2>&1 &
HTTP_PID=$!

# Start Streamlit backend
"$PYTHON" -m streamlit run app.py --server.port 8501 --server.headless true >/dev/null 2>&1 &
STREAMLIT_PID=$!

# Wait for background processes
wait "$HTTP_PID" "$STREAMLIT_PID" 2>/dev/null || true
cleanup
