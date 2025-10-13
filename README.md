# Orca Motor Control GUI

A NiceGUI-based web application for controlling Orca Series motors with custom PID position control.

## Features

- **Simple Interface**: Web-based GUI accessible from any browser
- **Two Motor Modes**: Sleep Mode and Force Mode only (Position/Kinematic/Haptic modes not used)
- **Custom PID Control**: Position control implemented using python-control library with custom PID
- **Real-time Telemetry**: Live monitoring of position, force, power, temperature, and voltage
- **Interactive Plots**: Real-time charts showing position and force over time
- **PID Tuning**: Adjust Kp, Ki, Kd gains on-the-fly
- **Safety Features**: Emergency stop button and force/position limits

## Control Modes

1. **Sleep Mode**: Motor is passive, no force control active
2. **Force Direct**: Direct force control - manually set force output in Newtons
3. **Position**: Position control using custom PID controller (outputs force commands)

## Architecture

```
motor_gui/
├── main.py              # Entry point - launches the GUI
├── app.py               # NiceGUI interface implementation
├── motor_controller.py  # Motor control logic and threading
├── pid_controller.py    # PID controller using python-control
└── requirements.txt     # Python dependencies
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Git (for cloning repositories)
- pip (Python package manager)

### Directory Structure

The motor_gui application expects the following directory structure:

```
YourWorkspace/
├── motor_gui/          # This application
├── pyorcasdk/          # Python SDK (required for local install)
├── orcaSDK/            # Optional: C++ SDK source
└── orcaSDK_tutorials/  # Optional: Tutorials
```

### Installation Steps

#### Option 1: Automated Installation (Recommended)

**On Linux/Mac:**
```bash
cd motor_gui
./install.sh
```

**On Windows:**
```cmd
cd motor_gui
install.bat
```

The install script will:
1. Check for Python 3.8+
2. Optionally create a virtual environment
3. Look for pyorcasdk in the parent directory
4. Initialize git submodules for pyorcasdk if needed
5. Install pyorcasdk from local source
6. Install all other dependencies (nicegui, control, numpy, plotly, pyserial)

#### Option 2: Manual Installation

1. **Clone Required Repositories** (if not already done):
   ```bash
   # Navigate to your workspace directory
   cd /path/to/YourWorkspace

   # Clone pyorcasdk (required)
   git clone https://github.com/IrisDynamics/pyorcasdk.git
   cd pyorcasdk
   git submodule update --init --recursive
   cd ..
   ```

2. **Install pyorcasdk**:
   ```bash
   pip install ./pyorcasdk
   ```

3. **Install Motor GUI Dependencies**:
   ```bash
   cd motor_gui
   pip install nicegui control numpy plotly pyserial
   ```

4. **Verify Installation**:
   ```bash
   python -c "import pyorcasdk; print('pyorcasdk installed successfully')"
   python -c "import nicegui; print('nicegui installed successfully')"
   ```

### Troubleshooting Installation

**Issue: "pyorcasdk build failed"**
- Ensure git submodules are initialized: `cd pyorcasdk && git submodule update --init --recursive`
- On Windows, ensure you have Visual Studio Build Tools installed
- On Linux, ensure you have build-essential: `sudo apt-get install build-essential cmake`

**Issue: "No module named 'pyorcasdk'"**
- Install from local directory: `pip install /path/to/pyorcasdk`
- Verify the directory structure matches the expected layout above

**Issue: "control library not found"**
- The package name is `control` not `python-control` on PyPI
- Install explicitly: `pip install control`

## Usage

1. **Launch the Application**:
   ```bash
   python main.py
   ```

   The application will automatically open in your default browser at `http://localhost:8080`

2. **Connect to Motor**:
   - Enter the serial port number (e.g., 3 for COM3 on Windows, or the port number on Linux)
   - Set baud rate (default: 1000000 for high-speed communication)
   - Click "Connect"

3. **Start Control Loop**:
   - Click "Start Control Loop" to begin the motor communication thread
   - The control loop runs at 100 Hz by default

4. **Control the Motor**:

   **Sleep Mode**:
   - Select "Sleep" from the mode dropdown
   - Motor will be passive and can be moved manually

   **Force Control**:
   - Select "Force Direct" from the mode dropdown
   - Enter desired force in Newtons
   - Click "Set Force"

   **Position Control**:
   - Select "Position" from the mode dropdown
   - Enter desired position in millimeters
   - Click "Set Position"
   - The PID controller will calculate the necessary force to reach the target

5. **Tune PID Controller**:
   - Adjust Kp (proportional), Ki (integral), and Kd (derivative) gains
   - Click "Update PID" to apply changes
   - Monitor the position plot to see the effect
   - Start with: Kp=0.1, Ki=0.01, Kd=0.005

6. **Monitor Telemetry**:
   - Current position and force displayed prominently
   - Power consumption, temperature, voltage, and errors shown in real-time
   - Position and force plots update continuously

7. **Emergency Stop**:
   - Click "EMERGENCY STOP" to immediately switch to Sleep Mode
   - Use this if the motor behaves unexpectedly

## PID Tuning Guide

The PID controller converts position error into force commands:

- **Kp (Proportional)**: Response to current error
  - Higher Kp = faster response, but may cause oscillation
  - Start with 0.1 and adjust

- **Ki (Integral)**: Eliminates steady-state error
  - Higher Ki = faster elimination of offset, but may cause overshoot
  - Start with 0.01 and adjust carefully
  - Set to 0 if not needed

- **Kd (Derivative)**: Dampens oscillations
  - Higher Kd = more damping, smoother response
  - Start with 0.005 and adjust
  - Can be sensitive to noise

### Tuning Process:
1. Start with Kp only (Ki=0, Kd=0)
2. Increase Kp until you get reasonable tracking with some oscillation
3. Add Kd to reduce oscillation
4. Add Ki if there's a steady-state position error
5. Fine-tune all three gains

## Configuration

### Control Loop Rate
Edit `motor_controller.py` line 44 to change the update rate:
```python
def __init__(self, name: str = "OrcaMotor", update_rate_hz: float = 100.0):
```

### Force Limits
Edit `pid_controller.py` lines 14-15 to change force limits:
```python
max_output: float = 50000.0  # Maximum force output in mN
min_output: float = -50000.0  # Minimum force output in mN
```

### Default PID Parameters
Edit `motor_controller.py` lines 60-66 to change default PID gains:
```python
pid_params = PIDParameters(
    kp=0.1,
    ki=0.01,
    kd=0.005,
    ...
)
```

## Safety Notes

- Always use the emergency stop if the motor behaves unexpectedly
- Start with low PID gains and increase gradually
- Monitor motor temperature and power consumption
- Ensure the motor has free travel range before enabling position control
- The motor will stop if communication is interrupted (safety feature)

## Troubleshooting

**Cannot connect to motor**:
- Verify correct serial port number
- Check cable connections
- Ensure no other program is using the serial port

**Position control oscillates**:
- Reduce Kp gain
- Increase Kd gain
- Check for mechanical issues (friction, binding)

**Position control too slow**:
- Increase Kp gain
- Add some Ki gain

**Motor overheats**:
- Reduce maximum force limits
- Lower PID gains
- Check for mechanical resistance

## Technical Details

- **Communication**: RS-422 serial at 1 Mbaud (configurable)
- **Control Frequency**: 100 Hz (configurable)
- **Position Units**: Micrometers (displayed as mm in GUI)
- **Force Units**: Millinewtons (displayed as N in GUI)
- **PID Implementation**: Discrete-time with anti-windup
- **Threading**: Separate thread for motor control loop

## License

This project uses the Orca SDK which is licensed separately by Iris Dynamics.
