# Complete setup script for Orca Motor Control GUI
# Run this with: powershell -ExecutionPolicy Bypass -File setup.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Orca Motor Control GUI - Complete Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python
Write-Host "Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = & python --version 2>&1
    Write-Host "Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python not found!" -ForegroundColor Red
    Write-Host "Install Python 3.8+ from: https://www.python.org/downloads/" -ForegroundColor Red
    pause
    exit 1
}

# Check Git
Write-Host "Checking Git installation..." -ForegroundColor Yellow
try {
    $gitVersion = & git --version 2>&1
    Write-Host "Found: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Git not found!" -ForegroundColor Red
    Write-Host "Install Git from: https://git-scm.com/download/win" -ForegroundColor Red
    pause
    exit 1
}

# Get parent directory
$parentDir = Split-Path -Parent (Get-Location)
$pyorcaPath = Join-Path $parentDir "pyorcasdk"

# Clone pyorcasdk if needed
if (-Not (Test-Path $pyorcaPath)) {
    Write-Host ""
    Write-Host "Cloning pyorcasdk..." -ForegroundColor Yellow
    Push-Location $parentDir
    git clone https://github.com/IrisDynamics/pyorcasdk.git
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to clone pyorcasdk" -ForegroundColor Red
        Pop-Location
        pause
        exit 1
    }
    Pop-Location
    Write-Host "pyorcasdk cloned successfully" -ForegroundColor Green
} else {
    Write-Host "pyorcasdk already exists" -ForegroundColor Green
}

# Initialize submodules
Write-Host ""
Write-Host "Initializing git submodules..." -ForegroundColor Yellow
Push-Location $pyorcaPath
git submodule update --init --recursive
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to initialize submodules" -ForegroundColor Red
    Pop-Location
    pause
    exit 1
}
Pop-Location
Write-Host "Submodules initialized" -ForegroundColor Green

# Create venv
Write-Host ""
if (Test-Path "venv") {
    Write-Host "Virtual environment already exists" -ForegroundColor Green
} else {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to create virtual environment" -ForegroundColor Red
        pause
        exit 1
    }
    Write-Host "Virtual environment created" -ForegroundColor Green
}

# Install pyorcasdk
Write-Host ""
Write-Host "Installing pyorcasdk..." -ForegroundColor Yellow
& ".\venv\Scripts\pip.exe" install $pyorcaPath
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install pyorcasdk" -ForegroundColor Red
    pause
    exit 1
}
Write-Host "pyorcasdk installed" -ForegroundColor Green

# Install other dependencies
Write-Host ""
Write-Host "Installing core packages..." -ForegroundColor Yellow
& ".\venv\Scripts\pip.exe" install numpy scipy matplotlib control plotly pyserial
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install core packages" -ForegroundColor Red
    pause
    exit 1
}

Write-Host ""
Write-Host "Installing NiceGUI dependencies..." -ForegroundColor Yellow
& ".\venv\Scripts\pip.exe" install Pygments aiofiles aiohttp certifi docutils fastapi h11 httpx ifaddr itsdangerous jinja2 markdown2 orjson python-engineio python-multipart python-socketio starlette typing-extensions uvicorn
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install NiceGUI dependencies" -ForegroundColor Red
    pause
    exit 1
}

Write-Host ""
Write-Host "Installing NiceGUI..." -ForegroundColor Yellow
& ".\venv\Scripts\pip.exe" install --no-deps nicegui
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install NiceGUI" -ForegroundColor Red
    pause
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Installation Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "To run the application:" -ForegroundColor Cyan
Write-Host "  .\venv\Scripts\python.exe main.py" -ForegroundColor White
Write-Host ""
Write-Host "Or create a shortcut with:" -ForegroundColor Cyan
Write-Host "  powershell -ExecutionPolicy Bypass -File run.ps1" -ForegroundColor White
Write-Host ""

# Ask if they want to run now
$run = Read-Host "Run the application now? (y/N)"
if ($run -eq "y" -or $run -eq "Y") {
    Write-Host ""
    Write-Host "Starting application..." -ForegroundColor Green
    & ".\venv\Scripts\python.exe" main.py
}
