#!/bin/bash
# Kill Pathway RAG Server
# Usage: ./kill_server.sh

PORT=8745
echo "Killing Pathway server on port $PORT..."

# Find and kill process on port 8745
PID=$(lsof -ti:$PORT 2>/dev/null)

if [ -z "$PID" ]; then
    echo "No process found on port $PORT"
else
    kill -9 $PID 2>/dev/null
    echo "Killed process $PID on port $PORT"
fi

# Also try killing by process name
pkill -f "python3.*run-server-gdrive.py" 2>/dev/null
echo "Done."

