@echo off
REM Installation script for Orca Motor Control GUI (Windows)

echo Installing Orca Motor Control GUI dependencies...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)

echo Python found:
python --version
echo.

REM Check if we're in the motor_gui directory
if not exist "main.py" (
    echo Error: Please run this script from the motor_gui directory.
    pause
    exit /b 1
)
if not exist "requirements.txt" (
    echo Error: Please run this script from the motor_gui directory.
    pause
    exit /b 1
)

REM Ask about virtual environment
set "create_venv=n"
set /p create_venv="Create a virtual environment? (recommended) [y/N]: "

if /i "%create_venv%"=="y" (
    echo.
    echo Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo Error: Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Install pyorcasdk from parent directory if available
if exist "..\pyorcasdk" (
    echo.
    echo Found pyorcasdk in parent directory. Installing from local source...

    REM Initialize git submodules if needed
    cd ..\pyorcasdk
    if exist ".git" (
        echo Initializing git submodules for pyorcasdk...
        git submodule update --init --recursive
    )
    cd ..\motor_gui

    REM Install pyorcasdk
    echo Installing pyorcasdk...
    pip install ..\pyorcasdk\
    if %errorlevel% neq 0 (
        echo Error: Failed to install pyorcasdk from local directory.
        pause
        exit /b 1
    )
) else (
    echo.
    echo Warning: pyorcasdk directory not found at ..\pyorcasdk
    echo Will attempt to install from PyPI (may fail if not published)...
)

REM Install remaining dependencies
echo.
echo Installing remaining dependencies...
pip install nicegui control numpy plotly pyserial

if %errorlevel% neq 0 (
    echo Error: Installation failed. Please check the error messages above.
    pause
    exit /b 1
)

REM Success message
echo.
echo ========================================
echo Installation complete!
echo ========================================
echo.

if /i "%create_venv%"=="y" (
    echo To run the application:
    echo   1. Activate the virtual environment:
    echo      venv\Scripts\activate.bat
    echo   2. Run the application:
    echo      python main.py
) else (
    echo To run the application:
    echo   python main.py
)

echo.
echo For help, see README.md
echo.
pause
