# Deployment Checklist for Target Machine

This guide helps you deploy the Motor Control GUI on a new machine.

## Before You Start

Ensure you have:
- [ ] Python 3.8 or higher installed
- [ ] Git installed
- [ ] USB-to-RS422 adapter drivers installed (if applicable)
- [ ] Orca motor powered and connected via RS422

## Step 1: Clone Repositories

```bash
# Create a workspace directory
mkdir OrcaWorkspace
cd OrcaWorkspace

# Clone the Python SDK (required)
git clone https://github.com/IrisDynamics/pyorcasdk.git

# Clone your motor_gui application
# (copy motor_gui folder here or git clone if you have a repo)
```

Your directory structure should look like:
```
OrcaWorkspace/
├── pyorcasdk/
└── motor_gui/
```

## Step 2: Run Installation Script

**On Windows:**
```cmd
cd motor_gui
install.bat
```

**On Linux/Mac:**
```bash
cd motor_gui
chmod +x install.sh
./install.sh
```

When prompted, choose "y" to create a virtual environment (recommended).

## Step 3: Verify Installation

```bash
# If using virtual environment, activate it first:
# Windows: venv\Scripts\activate.bat
# Linux/Mac: source venv/bin/activate

# Test imports
python -c "import pyorcasdk; print('pyorcasdk OK')"
python -c "import nicegui; print('nicegui OK')"
python -c "import control; print('control OK')"
```

## Step 4: Connect Motor and Test

1. **Connect the motor** via USB-to-RS422 adapter
2. **Power on the motor**
3. **Launch the application**:
   ```bash
   python main.py
   ```
4. **In the browser**:
   - Click "Refresh Ports" if motor not shown
   - Select your motor's serial port from dropdown
   - Click "Connect"
   - Click "Start Control Loop"
   - Test Sleep mode first (motor should be free to move)

## Troubleshooting

### No serial ports detected
- **Windows**: Check Device Manager for COM ports
  - Install FTDI or CH340 drivers if needed
- **Linux**: Check `/dev/ttyUSB*` or `/dev/ttyACM*`
  - Add user to dialout group: `sudo usermod -a -G dialout $USER` (then log out/in)
- **Mac**: Check `/dev/tty.usbserial*`

### Connection fails
- Verify baud rate is 1000000 (1M)
- Check cable connections
- Try different COM port
- Power cycle the motor

### Motor behaves erratically
- Check motor power supply (voltage/current)
- Reduce PID gains (start with Kp=0.05, Ki=0, Kd=0)
- Ensure mechanical system has no binding
- Check for proper cable shielding/grounding

### Application won't start
- Check Python version: `python --version` (needs 3.8+)
- Reinstall dependencies: `pip install --force-reinstall nicegui`
- Check for port conflicts (close other apps using port 8080)

## Production Deployment Notes

### Running as a Service (Linux)

Create `/etc/systemd/system/motor-gui.service`:
```ini
[Unit]
Description=Orca Motor Control GUI
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/path/to/OrcaWorkspace/motor_gui
ExecStart=/path/to/venv/bin/python main.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl daemon-reload
sudo systemctl enable motor-gui
sudo systemctl start motor-gui
```

### Running at Startup (Windows)

1. Create a batch file `start_motor_gui.bat`:
   ```cmd
   @echo off
   cd C:\path\to\OrcaWorkspace\motor_gui
   call venv\Scripts\activate.bat
   python main.py
   ```

2. Place shortcut in Startup folder:
   - Press `Win+R`, type `shell:startup`
   - Create shortcut to `start_motor_gui.bat`

### Network Access

To allow access from other devices on your network:

Edit `main.py`, change:
```python
ui.run(
    title='Orca Motor Control',
    port=8080,
    host='0.0.0.0',  # Add this line
    reload=False,
    show=True
)
```

Then access from other devices at: `http://<target-machine-ip>:8080`

## Safety Reminders

- [ ] Always start in Sleep mode when first connecting
- [ ] Test with low PID gains before increasing
- [ ] Keep Emergency Stop button visible
- [ ] Monitor motor temperature during operation
- [ ] Ensure mechanical system has proper end stops
- [ ] Document your final PID tuning values

## Support

For issues with:
- **orcaSDK**: Check [IrisDynamics documentation](https://irisdynamics.com/downloads)
- **Motor hardware**: Contact IrisDynamics support
- **This application**: Check README.md and motor_controller.py comments
