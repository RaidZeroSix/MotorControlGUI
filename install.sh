#!/bin/bash
# Installation script for Orca Motor Control GUI

echo "Installing Orca Motor Control GUI dependencies..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "Python 3 found: $(python3 --version)"

# Check if we're in the motor_gui directory
if [ ! -f "main.py" ] || [ ! -f "requirements.txt" ]; then
    echo "Error: Please run this script from the motor_gui directory."
    exit 1
fi

# Create virtual environment (optional but recommended)
read -p "Create a virtual environment? (recommended) [y/N]: " create_venv
if [[ $create_venv =~ ^[Yy]$ ]]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Install pyorcasdk from parent directory if available
if [ -d "../pyorcasdk" ]; then
    echo ""
    echo "Found pyorcasdk in parent directory. Installing from local source..."

    # Initialize git submodules if needed
    cd ../pyorcasdk
    if [ -d ".git" ]; then
        echo "Initializing git submodules for pyorcasdk..."
        git submodule update --init --recursive
    fi
    cd ../motor_gui

    # Install pyorcasdk
    pip install ../pyorcasdk/
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install pyorcasdk from local directory."
        exit 1
    fi
else
    echo "Warning: pyorcasdk directory not found at ../pyorcasdk"
    echo "Will attempt to install from PyPI (may fail if not published)..."
fi

# Install remaining dependencies
echo ""
echo "Installing remaining dependencies from requirements.txt..."
pip install nicegui control numpy plotly pyserial

if [ $? -eq 0 ]; then
    echo ""
    echo "Installation complete!"
    echo ""
    echo "To run the application:"
    if [[ $create_venv =~ ^[Yy]$ ]]; then
        echo "  1. Activate the virtual environment: source venv/bin/activate"
        echo "  2. Run: python3 main.py"
    else
        echo "  python3 main.py"
    fi
else
    echo "Error: Installation failed. Please check the error messages above."
    exit 1
fi
