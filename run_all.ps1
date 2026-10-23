# Launch the full SuryaIQ stack on Windows: FastAPI (:8000) + Streamlit (:8530)
$ErrorActionPreference = "Stop"

Write-Host "Starting FastAPI on http://localhost:8000 ..." -ForegroundColor Cyan
Start-Process -FilePath "python" -ArgumentList "-m uvicorn api.main:app --host 127.0.0.1 --port 8000"

Start-Sleep -Seconds 3

try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -TimeoutSec 3
    Write-Host "FastAPI Health OK: $($health | ConvertTo-Json -Compress)" -ForegroundColor Green
} catch {
    Write-Host "Warning: Could not connect to FastAPI health check yet." -ForegroundColor Yellow
}

Write-Host "Starting Streamlit dashboard on http://localhost:8530 ..." -ForegroundColor Cyan
python -m streamlit run app/streamlit_app.py --server.port 8530
