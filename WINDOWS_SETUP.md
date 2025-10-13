# Windows Setup Guide

## Quick Start

If you just want to get running ASAP, here's the deal:

```powershell
git clone https://github.com/RaidZeroSix/MotorControlGUI.git
cd MotorControlGUI
powershell -ExecutionPolicy Bypass -File setup.ps1
```

The script will detect if you're missing build tools and guide you through fixing it.

---

## Prerequisites (One-Time Setup)

pyorcasdk is a Python wrapper around C++ code, which means it needs to **compile C++ code** on your machine. Windows doesn't come with C++ build tools by default.

### Option 1: Install Visual Studio Build Tools (Recommended)

**This is a ~7 GB download but it's the most reliable approach.**

1. Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Run the installer
3. Select **"Desktop development with C++"** workload
4. Click Install (takes 10-20 minutes)
5. **Restart your terminal** after installation
6. Run `setup.ps1` again

### Option 2: Use Existing Visual Studio

If you already have Visual Studio 2019 or 2022 installed with C++ tools:

1. Search "Developer PowerShell for VS" in Start Menu
2. Launch that instead of regular PowerShell
3. Navigate to MotorControlGUI directory
4. Run `setup.ps1`

---

## What the setup.ps1 Script Does

1. ✅ Checks Python 3.8+ is installed
2. ✅ Checks Git is installed
3. ✅ **Checks for C++ Build Tools** (warns if missing)
4. ✅ Clones pyorcasdk SDK to parent directory
5. ✅ Initializes git submodules
6. ✅ Creates Python virtual environment
7. ✅ Builds and installs pyorcasdk from source
8. ✅ Installs all other dependencies
9. ✅ Offers to launch the application

---

## Troubleshooting

### "nmake not found" or "CMAKE_CXX_COMPILER not set"

**Cause**: C++ Build Tools not installed or not in PATH.

**Fix**:
- Install Visual Studio Build Tools (see Option 1 above)
- OR use Developer PowerShell for VS (see Option 2 above)

### "Python not found"

**Cause**: Python not installed or not in PATH.

**Fix**: Install Python 3.8+ from https://www.python.org/downloads/
- During installation, **check "Add Python to PATH"**

### "Git not found"

**Cause**: Git not installed or not in PATH.

**Fix**: Install Git from https://git-scm.com/download/win

### Build succeeds but application won't run

**Cause**: You're running Python from outside the virtual environment.

**Fix**: Always use the venv Python:
```powershell
.\venv\Scripts\python.exe main.py
```

---

## After Successful Installation

Run the application:
```powershell
.\venv\Scripts\python.exe main.py
```

Or create a shortcut on your desktop that runs:
```powershell
powershell -ExecutionPolicy Bypass -Command "cd 'C:\path\to\MotorControlGUI'; .\venv\Scripts\python.exe main.py"
```

---

## Why Do We Need C++ Build Tools?

The Orca SDK is written in C++ for performance (it needs to run at 1000+ Hz). The Python bindings are created using **pybind11**, which generates a `.pyd` file (Python extension module) from the C++ source code.

This compilation happens on **your machine** when you run `pip install pyorcasdk`, which is why you need:
- **CMake**: Build system generator
- **nmake**: Microsoft's make tool
- **cl.exe**: Microsoft C++ compiler
- **Python headers**: Included with Python installation

All of these come with Visual Studio Build Tools.

---

## Alternative: Pre-built Wheels (Not Yet Available)

If IrisDynamics publishes pre-built wheels to PyPI in the future, you won't need build tools. But for now, compiling from source is the only option.
