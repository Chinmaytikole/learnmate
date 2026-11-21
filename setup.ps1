# setup.ps1 — Windows PowerShell setup script (alternative to make setup)
# Run from the learnmate/ directory:
#   powershell -ExecutionPolicy Bypass -File setup.ps1

$ErrorActionPreference = "Stop"

Write-Host "=== LearnMate Setup ===" -ForegroundColor Cyan

# 1. Create virtual environment if not already present
if (-not (Test-Path ".venv")) {
    Write-Host "[1/4] Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
} else {
    Write-Host "[1/4] Virtual environment already exists." -ForegroundColor Green
}

# 2. Activate and install dependencies
Write-Host "[2/4] Installing Python dependencies (this may take 5-10 minutes)..." -ForegroundColor Yellow
& .\.venv\Scripts\pip install -r requirements.txt

# 3. Pull Ollama models (requires Ollama running)
Write-Host "[3/4] Pulling Ollama models (requires 'ollama serve' running)..." -ForegroundColor Yellow
Write-Host "  Pulling llama3.2:3b (generator)..." -ForegroundColor Gray
ollama pull llama3.2:3b
Write-Host "  Pulling llama3.1:8b (question generation)..." -ForegroundColor Gray
ollama pull llama3.1:8b

# 4. Verify with tests
Write-Host "[4/4] Running Phase 0 tests..." -ForegroundColor Yellow
& .\.venv\Scripts\python -m pytest tests/test_utils.py -v

Write-Host ""
Write-Host "=== Setup complete! ===" -ForegroundColor Green
Write-Host "Next: run Phase 1 with:" -ForegroundColor Cyan
Write-Host "  .\.venv\Scripts\python scripts/make_squad_splits.py --config configs/default.yaml"
