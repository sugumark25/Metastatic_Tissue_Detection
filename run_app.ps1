Write-Host "Checking dependencies..."
pip install -r backend/requirements.txt

Write-Host "Checking data..."
if (-not (Test-Path "data/data_sample")) {
    Write-Host "Error: Data not found in data/data_sample. Please ensure data is present."
}

Write-Host "Training model (this may take a while)..."
if (-not (Test-Path "models/model.pth")) {
    Set-Location backend
    python train.py
    Set-Location ..
}

Write-Host "Starting Server..."
Start-Process -FilePath "uvicorn" -ArgumentList "main:app --reload --app-dir backend" -NoNewWindow
Write-Host "Server started at http://127.0.0.1:8000"

Write-Host "Opening Frontend..."
Start-Process "frontend/index.html"
