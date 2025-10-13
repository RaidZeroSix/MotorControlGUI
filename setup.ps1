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

# Check for C++ Build Tools (required for pyorcasdk compilation)
Write-Host ""
Write-Host "Checking for C++ Build Tools..." -ForegroundColor Yellow
$vsWhereAvailable = $false
$buildToolsFound = $false

# Try to find Visual Studio using vswhere
$vsWherePath = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vsWherePath) {
    $vsWhereAvailable = $true
    $vsInstalls = & $vsWherePath -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($vsInstalls) {
        $buildToolsFound = $true
        Write-Host "Found Visual Studio C++ Build Tools" -ForegroundColor Green
    }
}

# Alternative check: Look for cl.exe (C++ compiler) in PATH
if (-not $buildToolsFound) {
    try {
        $clVersion = & cl.exe /? 2>&1 | Select-Object -First 1
        if ($clVersion) {
            $buildToolsFound = $true
            Write-Host "Found C++ compiler (cl.exe)" -ForegroundColor Green
        }
    } catch {
        # cl.exe not in PATH
    }
}

if (-not $buildToolsFound) {
    Write-Host ""
    Write-Host "WARNING: C++ Build Tools not detected!" -ForegroundColor Red
    Write-Host ""
    Write-Host "pyorcasdk requires C++ compilation. You have two options:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Option 1 - Install Visual Studio Build Tools (RECOMMENDED):" -ForegroundColor Cyan
    Write-Host "  1. Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/" -ForegroundColor White
    Write-Host "  2. Run installer and select 'Desktop development with C++'" -ForegroundColor White
    Write-Host "  3. Install (requires ~6-7 GB)" -ForegroundColor White
    Write-Host "  4. Restart this terminal and run setup.ps1 again" -ForegroundColor White
    Write-Host ""
    Write-Host "Option 2 - Use Developer PowerShell:" -ForegroundColor Cyan
    Write-Host "  If you already have Visual Studio installed:" -ForegroundColor White
    Write-Host "  1. Launch 'Developer PowerShell for VS' from Start Menu" -ForegroundColor White
    Write-Host "  2. Navigate to this directory" -ForegroundColor White
    Write-Host "  3. Run setup.ps1 again" -ForegroundColor White
    Write-Host ""

    $continue = Read-Host "Continue anyway and attempt installation? (y/N)"
    if ($continue -ne "y" -and $continue -ne "Y") {
        Write-Host ""
        Write-Host "Setup cancelled. Install build tools and run again." -ForegroundColor Yellow
        pause
        exit 0
    }
    Write-Host ""
    Write-Host "Proceeding with installation (may fail)..." -ForegroundColor Yellow
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
