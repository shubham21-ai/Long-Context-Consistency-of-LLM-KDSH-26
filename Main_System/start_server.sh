#!/bin/bash
# Start Pathway RAG Server
# Usage: ./start_server.sh

cd "$(dirname "$0")/Pathway_code"
echo "Starting Pathway RAG Server..."
echo "Server will run on http://127.0.0.1:8745"
echo "Press Ctrl+C to stop"
echo ""
python3 run-server-gdrive.py

