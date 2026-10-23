#!/usr/bin/env bash
# Launch the full SuryaIQ stack: FastAPI (orchestration) + Streamlit (dashboard).
# Matches the architecture diagram: Dashboard -> FastAPI -> LangGraph.
set -e
cd "$(dirname "$0")"
source .venv/bin/activate

echo "Starting FastAPI on :8000 ..."
pkill -f "uvicorn api.main" 2>/dev/null || true
nohup .venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000 >/tmp/suryaiq_api.log 2>&1 &
sleep 4
curl -s http://localhost:8000/health && echo

echo "Starting Streamlit dashboard on :8530 ..."
pkill -f "streamlit run app/streamlit_app.py" 2>/dev/null || true
.venv/bin/streamlit run app/streamlit_app.py --server.port 8530
