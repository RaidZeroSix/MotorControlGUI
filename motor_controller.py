"""
Motor Controller Module

Manages the Orca motor communication and control loop.
Uses Force Mode and Sleep Mode only, with custom PID position control.
"""

import threading
import time
import json
import os
import sys
import platform
from typing import Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

from pyorcasdk import Actuator, MotorMode, OrcaError
from pid_controller import PIDController, PIDParameters
import numpy as np


class MotorCommunicationError(Exception):
    """Exception raised for motor communication errors that should trigger reconnection"""
    pass

# Platform-specific imports for process priority
if platform.system() == 'Windows':
    try:
        import win32process
        import win32api
        WINDOWS_PRIORITY_AVAILABLE = True
    except ImportError:
        WINDOWS_PRIORITY_AVAILABLE = False
        print("Warning: pywin32 not installed. Run 'pip install pywin32' for better performance on Windows.")
else:
    WINDOWS_PRIORITY_AVAILABLE = False


class ControlMode(Enum):
    """Control modes for the application"""
    SLEEP = "sleep"
    FORCE_DIRECT = "force_direct"  # Direct force control
    POSITION = "position"  # Position control using PID
    SHOCK_PROFILE = "shock_profile"  # Shock profile state machine


class ShockState(Enum):
    """States for shock profile execution"""
    IDLE = "idle"
    ACCELERATE = "accelerate"
    DECELERATE = "decelerate"
    STABILIZE = "stabilize"
    WAIT = "wait"
    HOMING = "homing"


@dataclass
class MotorState:
    """Current state of the motor"""
    position_um: int = 0
    force_mN: int = 0
    power_W: int = 0
    temperature_C: int = 0
    voltage_mV: int = 0
    errors: int = 0
    mode: MotorMode = MotorMode.SleepMode
    control_mode: ControlMode = ControlMode.SLEEP
    connected: bool = False
    running: bool = False
    shock_state: ShockState = ShockState.IDLE
    thermal_pause: bool = False


@dataclass
class MotorCommand:
    """Commands to send to the motor"""
    control_mode: ControlMode = ControlMode.SLEEP
    force_setpoint_mN: float = 0.0
    position_setpoint_um: float = 0.0


@dataclass
class ShockProfileParameters:
    """Parameters for shock profile execution"""
    name: str = "Default Profile"
    accel_force_N: float = 150.0  # Acceleration force in Newtons
    decel_force_N: float = -180.0  # Deceleration/reverse force in Newtons
    switch_position_mm: float = 60.0  # Position to switch from accel to decel
    end_position_mm: float = 100.0  # Bidirectional crossing threshold
    repetitions: int = 1  # Number of times to repeat (1 = single run)
    wait_time_s: float = 2.0  # Wait time after stabilize before homing
    homing_force_N: float = 50.0  # Force to apply when returning home


class StateEstimator:
    """
    Kalman filter that estimates position and velocity between measurements.
    Uses force measurements to predict motion during communication gaps.
    Fills in spatial resolution gaps at high speeds with variable sample rates.
    """

    # System masses
    MASS_ANVIL_KG = 8.6  # kg - anvil mass alone
    MASS_STRIKER_KG = 4.5  # kg - striker mass
    MASS_TOTAL_KG = MASS_ANVIL_KG + MASS_STRIKER_KG  # 13.1 kg total

    def __init__(self, mass_kg=MASS_TOTAL_KG):
        """
        Initialize state estimator.

        Args:
            mass_kg: Initial mass (default: anvil + striker)
        """
        self.current_mass = mass_kg

        # State vector: [position_mm, velocity_mm_s]
        self.x = np.array([0.0, 0.0])

        # State covariance (uncertainty in our estimate)
        self.P = np.array([
            [1.0, 0.0],      # Low position uncertainty initially
            [0.0, 100.0]     # Higher velocity uncertainty initially
        ])

        # Process noise (accounts for model uncertainty, friction variations)
        self.Q_base = np.array([
            [0.1, 0.0],       # Position process noise
            [0.0, 200.0]      # Velocity process noise (friction uncertainty)
        ])
        self.Q = self.Q_base.copy()

        # Measurement noise (motor's position sensor has 1μm resolution)
        self.R = np.array([[0.01]])  # 0.01mm² variance

        # Friction estimate (learned over time)
        self.friction_N = 20.0
        self.friction_alpha = 0.02  # Slow adaptation rate

        # Diagnostics
        self.innovation_history = []

    def set_mass(self, mass_kg):
        """
        Update mass when system configuration changes.
        Call this when switching between acceleration and deceleration.

        Args:
            mass_kg: New mass value
        """
        self.current_mass = mass_kg

    def reset_state(self, position_mm=0.0, velocity_mm_s=0.0):
        """
        Reset state estimate (e.g., after zeroing position).

        Args:
            position_mm: Initial position
            velocity_mm_s: Initial velocity
        """
        self.x = np.array([position_mm, velocity_mm_s])
        self.P = np.array([
            [1.0, 0.0],
            [0.0, 100.0]
        ])

    def predict(self, force_commanded_N, dt):
        """
        Predict state forward using dynamics model.
        This fills in the blanks between sparse measurements!

        Uses kinematic equations:
            x_new = x_old + v*dt + 0.5*a*dt²
            v_new = v_old + a*dt
        where a = (F_commanded - F_friction) / m

        Args:
            force_commanded_N: Commanded force in Newtons
            dt: Time step since last update
        """
        if dt <= 0 or dt > 0.1:  # Sanity check
            return

        # Estimate acceleration from commanded force and friction
        # Friction opposes motion: F_friction = -friction_magnitude * sign(velocity)
        friction_force = -self.friction_N * np.sign(self.x[1]) if abs(self.x[1]) > 1.0 else 0
        a_mm_s2 = ((force_commanded_N - friction_force) / self.current_mass) * 1000.0

        # State transition matrix (kinematic equations)
        F = np.array([
            [1.0, dt],
            [0.0, 1.0]
        ])

        # Control input matrix (force affects acceleration)
        B = np.array([
            [0.5 * dt**2],
            [dt]
        ])

        # Predict state: x_new = F*x + B*a
        self.x = F @ self.x + B.flatten() * a_mm_s2

        # Predict covariance: P_new = F*P*F' + Q
        # Scale Q by dt (more uncertainty over longer predictions)
        Q_scaled = self.Q * (dt / 0.001)  # Normalized to 1ms
        self.P = F @ self.P @ F.T + Q_scaled

    def update(self, position_measured_mm):
        """
        Update state estimate with new position measurement.
        Corrects our prediction when we get new data from motor.

        Args:
            position_measured_mm: Measured position from motor
        """
        # Measurement matrix (we only measure position, not velocity)
        H = np.array([[1.0, 0.0]])

        # Innovation (measurement residual)
        y = position_measured_mm - H @ self.x
        self.innovation_history.append(abs(y[0]))

        # Keep last 100 innovations for diagnostics
        if len(self.innovation_history) > 100:
            self.innovation_history.pop(0)

        # Innovation covariance
        S = H @ self.P @ H.T + self.R

        # Kalman gain (how much to trust measurement vs prediction)
        K = self.P @ H.T / S

        # Update state estimate
        self.x = self.x + K.flatten() * y

        # Update covariance (uncertainty decreases after measurement)
        I = np.eye(2)
        self.P = (I - K @ H) @ self.P

        # Adapt friction estimate based on systematic errors
        if len(self.innovation_history) >= 10:
            avg_innovation = np.mean(self.innovation_history[-10:])
            if avg_innovation > 1.0:  # Systematic error > 1mm
                # Adjust friction slightly
                friction_adjustment = 0.5 * np.sign(y[0])
                self.friction_N += self.friction_alpha * friction_adjustment
                self.friction_N = max(0.0, min(self.friction_N, 100.0))  # Clamp

    def predict_future_state(self, force_commanded_N, time_ahead_s):
        """
        Predict where we'll be in the future (look-ahead for early commands).
        Does NOT modify internal state.

        Args:
            force_commanded_N: Force that will be commanded
            time_ahead_s: How far ahead to predict (seconds)

        Returns:
            Tuple of (predicted_position_mm, predicted_velocity_mm_s)
        """
        # Copy current state
        x_future = self.x.copy()

        # Simple forward prediction
        # Friction opposes motion: F_friction = -friction_magnitude * sign(velocity)
        friction_force = -self.friction_N * np.sign(x_future[1]) if abs(x_future[1]) > 1.0 else 0
        a_mm_s2 = ((force_commanded_N - friction_force) / self.current_mass) * 1000.0

        # Kinematic update
        x_future[0] += x_future[1] * time_ahead_s + 0.5 * a_mm_s2 * time_ahead_s**2
        x_future[1] += a_mm_s2 * time_ahead_s

        return x_future[0], x_future[1]

    def get_position_mm(self):
        """Get estimated position in mm"""
        return self.x[0]

    def get_velocity_mm_s(self):
        """Get estimated velocity in mm/s"""
        return self.x[1]

    def get_state(self):
        """Get full state vector [position_mm, velocity_mm_s]"""
        return self.x.copy()

    def get_uncertainty(self):
        """Get position and velocity uncertainties (standard deviations)"""
        return np.sqrt(np.diag(self.P))

    def increase_process_noise(self, factor=10.0):
        """Increase process noise during unpredictable phases (e.g., near collision)"""
        self.Q = self.Q_base * factor

    def reset_process_noise(self):
        """Reset to normal process noise"""
        self.Q = self.Q_base.copy()


class MotorController:
    """
    Main motor controller class.

    Manages motor communication, control loop, and PID position control.
    """

    def __init__(self, name: str = "OrcaMotor", update_rate_hz: float = 1000.0):
        """
        Initialize the motor controller.

        Args:
            name: Name for the actuator
            update_rate_hz: Control loop update rate in Hz (0 = maximum speed, no sleep)
        """
        self.actuator = Actuator(name)
        self.update_rate_hz = update_rate_hz
        self.update_period = 1.0 / update_rate_hz if update_rate_hz > 0 else 0

        # PID controller with default parameters
        pid_params = PIDParameters(
            kp=0.1,
            ki=0.01,
            kd=0.005,
            max_output=30000.0,
            min_output=-30000.0,
            sample_time=self.update_period
        )
        self.pid = PIDController(pid_params)

        # State
        self.state = MotorState()
        self.command = MotorCommand()

        # Shock profile parameters and state
        self.shock_params = ShockProfileParameters()
        self.shock_state = ShockState.IDLE
        self.shock_first_crossing = False
        self.shock_stabilize_position_um = 0
        self.shock_current_repetition = 0  # Current repetition count (0-based)
        self.shock_wait_start_time = 0.0  # Time when wait state started
        self.shock_thermal_pause = False  # Flag for thermal management pause

        # Profiles directory
        self.profiles_dir = Path("profiles")
        self.profiles_dir.mkdir(exist_ok=True)

        # State estimator for improved observability
        self.state_estimator = StateEstimator(mass_kg=StateEstimator.MASS_TOTAL_KG)

        # Timing for state estimation
        self.last_measurement_time = None
        self.last_force_command_N = 0.0

        # Look-ahead parameters (for early command sending)
        self.look_ahead_messages = 2  # Send command 2 messages early
        self.estimated_message_period_s = 0.0011  # 1.1ms @ 900Hz (updated dynamically)

        # Connection parameters for auto-reconnection
        self._connection_port = None
        self._connection_baud_rate = 1000000
        self._connection_interframe_delay = 80

        # Thread safety
        self._state_lock = threading.Lock()
        self._command_lock = threading.Lock()
        self._shock_lock = threading.Lock()

        # Control loop thread
        self._control_thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()

        # Callback for state updates (called from control thread)
        self.state_update_callback: Optional[Callable[[MotorState], None]] = None

    def connect(self, port, baud_rate: int = 1000000, interframe_delay: int = 80) -> bool:
        """
        Connect to the motor via serial port.

        Args:
            port: Serial port (int for COM port number or str for device path like '/dev/ttyUSB0')
            baud_rate: Baud rate for communication (default 1M for high-speed)
            interframe_delay: Interframe delay in microseconds

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Store connection parameters for auto-reconnection
            self._connection_port = port
            self._connection_baud_rate = baud_rate
            self._connection_interframe_delay = interframe_delay

            error = self.actuator.open_serial_port(port, baud_rate, interframe_delay)
            if error:
                print(f"Error opening serial port: {error.what()}")
                return False

            # Clear any existing errors
            self.actuator.clear_errors()

            # Enable streaming for high-speed communication
            self.actuator.enable_stream()

            # Set to sleep mode initially
            error = self.actuator.set_mode(MotorMode.SleepMode)
            if error:
                print(f"Error setting sleep mode: {error.what()}")
                return False

            with self._state_lock:
                self.state.connected = True
                self.state.mode = MotorMode.SleepMode
                self.state.control_mode = ControlMode.SLEEP

            return True

        except Exception as e:
            print(f"Exception during connection: {e}")
            return False

    def disconnect(self):
        """Disconnect from the motor"""
        self.stop_control_loop()

        try:
            # Set to sleep mode before disconnecting
            self.actuator.set_mode(MotorMode.SleepMode)
            self.actuator.disable_stream()
            self.actuator.close_serial_port()
        except Exception as e:
            print(f"Exception during disconnect: {e}")

        with self._state_lock:
            self.state.connected = False
            self.state.running = False

    def reconnect(self) -> bool:
        """
        Attempt to reconnect to the motor using stored connection parameters.
        Preserves shock profile state for seamless resumption.

        Returns:
            True if reconnection successful, False otherwise
        """
        if self._connection_port is None:
            print("No connection parameters stored, cannot reconnect")
            return False

        print(f"Attempting to reconnect to motor on port {self._connection_port}...")

        try:
            # Close existing connection if any
            try:
                self.actuator.close_serial_port()
            except:
                pass

            # Attempt reconnection
            error = self.actuator.open_serial_port(
                self._connection_port,
                self._connection_baud_rate,
                self._connection_interframe_delay
            )
            if error:
                print(f"Error reconnecting to serial port: {error.what()}")
                return False

            # Clear any existing errors
            self.actuator.clear_errors()

            # Enable streaming for high-speed communication
            self.actuator.enable_stream()

            # Restore motor to appropriate mode based on current control mode
            with self._state_lock:
                current_control_mode = self.state.control_mode

            if current_control_mode == ControlMode.SLEEP:
                error = self.actuator.set_mode(MotorMode.SleepMode)
                if error:
                    print(f"Error setting sleep mode: {error.what()}")
                    return False
            else:
                # Any active control mode uses force mode
                error = self.actuator.set_mode(MotorMode.ForceMode)
                if error:
                    print(f"Error setting force mode: {error.what()}")
                    return False

            with self._state_lock:
                self.state.connected = True

            print("Reconnection successful! Shock profile will resume where it left off.")
            return True

        except Exception as e:
            print(f"Exception during reconnection: {e}")
            return False

    def start_control_loop(self) -> bool:
        """
        Start the control loop thread.

        Returns:
            True if started successfully
        """
        if self._control_thread is not None and self._control_thread.is_alive():
            print("Control loop already running")
            return False

        if not self.state.connected:
            print("Not connected to motor")
            return False

        # Set process priority to high (Windows only)
        self._set_process_priority()

        self._stop_flag.clear()
        self._control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._control_thread.start()

        with self._state_lock:
            self.state.running = True

        return True

    def _set_process_priority(self):
        """Set process and thread priority for real-time performance (Windows)"""
        if WINDOWS_PRIORITY_AVAILABLE:
            try:
                # Get current process handle
                pid = win32api.GetCurrentProcessId()
                handle = win32api.OpenProcess(win32process.PROCESS_ALL_ACCESS, True, pid)

                # Set process priority to HIGH (not REALTIME to avoid system lockup)
                win32process.SetPriorityClass(handle, win32process.HIGH_PRIORITY_CLASS)
                print("Process priority set to HIGH")

            except Exception as e:
                print(f"Could not set process priority: {e}")
                print("Try running as Administrator for better performance")
        elif platform.system() == 'Windows':
            print("Install pywin32 for better performance: pip install pywin32")

    def stop_control_loop(self):
        """Stop the control loop thread"""
        if self._control_thread is None or not self._control_thread.is_alive():
            return

        self._stop_flag.set()
        self._control_thread.join(timeout=2.0)

        with self._state_lock:
            self.state.running = False

    def set_control_mode(self, mode: ControlMode):
        """
        Set the control mode.

        Args:
            mode: Desired control mode
        """
        with self._command_lock:
            self.command.control_mode = mode

        # Reset PID when switching to position mode
        if mode == ControlMode.POSITION:
            self.pid.reset()

    def set_force_setpoint(self, force_mN: float):
        """
        Set the force setpoint for direct force control.

        Args:
            force_mN: Force in millinewtons
        """
        with self._command_lock:
            self.command.force_setpoint_mN = force_mN

    def set_position_setpoint(self, position_um: float):
        """
        Set the position setpoint for position control.

        Args:
            position_um: Position in micrometers
        """
        with self._command_lock:
            self.command.position_setpoint_um = position_um

    def update_pid_parameters(self, kp: Optional[float] = None,
                             ki: Optional[float] = None,
                             kd: Optional[float] = None,
                             max_output: Optional[float] = None,
                             min_output: Optional[float] = None):
        """
        Update PID controller parameters.

        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            max_output: Maximum force output in mN
            min_output: Minimum force output in mN
        """
        self.pid.update_parameters(kp=kp, ki=ki, kd=kd, max_output=max_output, min_output=min_output)

    def get_state(self) -> MotorState:
        """
        Get a copy of the current motor state.

        Returns:
            Copy of the current MotorState
        """
        with self._state_lock:
            # Create a copy to avoid threading issues
            return MotorState(
                position_um=self.state.position_um,
                force_mN=self.state.force_mN,
                power_W=self.state.power_W,
                temperature_C=self.state.temperature_C,
                voltage_mV=self.state.voltage_mV,
                errors=self.state.errors,
                mode=self.state.mode,
                control_mode=self.state.control_mode,
                connected=self.state.connected,
                running=self.state.running,
                shock_state=self.shock_state
            )

    def emergency_stop(self):
        """Emergency stop - immediately switch to sleep mode"""
        self.set_control_mode(ControlMode.SLEEP)
        try:
            self.actuator.set_mode(MotorMode.SleepMode)
        except Exception as e:
            print(f"Error during emergency stop: {e}")

    def zero_position(self):
        """Zero the motor position at current location"""
        try:
            self.actuator.zero_position()
            # Reset state estimator when zeroing position
            self.state_estimator.reset_state(position_mm=0.0, velocity_mm_s=0.0)
            print("Position zeroed (estimator reset)")
        except Exception as e:
            print(f"Error zeroing position: {e}")

    def set_shock_profile_parameters(self, name: str = None, accel_force_N: float = None,
                                     decel_force_N: float = None, switch_position_mm: float = None,
                                     end_position_mm: float = None, repetitions: int = None,
                                     wait_time_s: float = None, homing_force_N: float = None):
        """Set shock profile parameters (only updates provided values)"""
        with self._shock_lock:
            if name is not None:
                self.shock_params.name = name
            if accel_force_N is not None:
                self.shock_params.accel_force_N = accel_force_N
            if decel_force_N is not None:
                self.shock_params.decel_force_N = decel_force_N
            if switch_position_mm is not None:
                self.shock_params.switch_position_mm = switch_position_mm
            if end_position_mm is not None:
                self.shock_params.end_position_mm = end_position_mm
            if repetitions is not None:
                self.shock_params.repetitions = repetitions
            if wait_time_s is not None:
                self.shock_params.wait_time_s = wait_time_s
            if homing_force_N is not None:
                self.shock_params.homing_force_N = homing_force_N

    def start_shock_profile(self):
        """Start shock profile execution"""
        with self._shock_lock:
            if self.shock_state == ShockState.IDLE:
                self.shock_state = ShockState.ACCELERATE
                self.shock_first_crossing = False
                self.shock_current_repetition = 0
                print(f"Shock profile started: {self.shock_params.repetitions} repetition(s)")
            else:
                print(f"Cannot start shock profile - current state: {self.shock_state}")

    def abort_shock_profile(self):
        """Abort shock profile execution"""
        with self._shock_lock:
            self.shock_state = ShockState.IDLE
            self.shock_first_crossing = False
            print("Shock profile aborted")

    def save_shock_profile(self, name: str = None) -> bool:
        """
        Save current shock profile parameters to disk.

        Args:
            name: Profile name (if None, uses current params.name)

        Returns:
            True if saved successfully
        """
        try:
            with self._shock_lock:
                profile_name = name if name else self.shock_params.name
                # Update name in params if provided
                if name:
                    self.shock_params.name = name

                # Convert to dict and save
                profile_dict = asdict(self.shock_params)

                # Sanitize filename
                safe_name = "".join(c for c in profile_name if c.isalnum() or c in (' ', '-', '_')).strip()
                filename = self.profiles_dir / f"{safe_name}.json"

                with open(filename, 'w') as f:
                    json.dump(profile_dict, f, indent=2)

                print(f"Profile saved: {filename}")
                return True

        except Exception as e:
            print(f"Error saving profile: {e}")
            return False

    def load_shock_profile(self, name: str) -> bool:
        """
        Load shock profile parameters from disk.

        Args:
            name: Profile name to load

        Returns:
            True if loaded successfully
        """
        try:
            # Sanitize filename
            safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()
            filename = self.profiles_dir / f"{safe_name}.json"

            if not filename.exists():
                print(f"Profile not found: {filename}")
                return False

            with open(filename, 'r') as f:
                profile_dict = json.load(f)

            with self._shock_lock:
                # Update parameters from dict
                self.shock_params = ShockProfileParameters(**profile_dict)

            print(f"Profile loaded: {filename}")
            return True

        except Exception as e:
            print(f"Error loading profile: {e}")
            return False

    def list_shock_profiles(self) -> list:
        """
        List all available shock profiles.

        Returns:
            List of profile names
        """
        try:
            profiles = []
            for file in self.profiles_dir.glob("*.json"):
                profiles.append(file.stem)
            return sorted(profiles)
        except Exception as e:
            print(f"Error listing profiles: {e}")
            return []

    def get_shock_profile_parameters(self) -> ShockProfileParameters:
        """Get a copy of current shock profile parameters"""
        with self._shock_lock:
            return ShockProfileParameters(
                name=self.shock_params.name,
                accel_force_N=self.shock_params.accel_force_N,
                decel_force_N=self.shock_params.decel_force_N,
                switch_position_mm=self.shock_params.switch_position_mm,
                end_position_mm=self.shock_params.end_position_mm,
                repetitions=self.shock_params.repetitions,
                wait_time_s=self.shock_params.wait_time_s,
                homing_force_N=self.shock_params.homing_force_N
            )

    def _control_loop(self):
        """Main control loop (runs in separate thread)"""
        print(f"Control loop started (target: {self.update_rate_hz if self.update_rate_hz > 0 else 'MAX'} Hz)")

        # Set thread priority to highest (Windows only)
        if WINDOWS_PRIORITY_AVAILABLE:
            try:
                import ctypes
                # Get current thread handle
                thread_handle = ctypes.windll.kernel32.GetCurrentThread()
                # Set to THREAD_PRIORITY_HIGHEST (not TIME_CRITICAL to avoid issues)
                THREAD_PRIORITY_HIGHEST = 2
                ctypes.windll.kernel32.SetThreadPriority(thread_handle, THREAD_PRIORITY_HIGHEST)
                print("Control thread priority set to HIGHEST")
            except Exception as e:
                print(f"Could not set thread priority: {e}")

        # Increase timer resolution on Windows for more accurate sleep
        if platform.system() == 'Windows':
            try:
                import ctypes
                # Request 1ms timer resolution
                ctypes.windll.winmm.timeBeginPeriod(1)
                print("Windows timer resolution set to 1ms")
            except Exception as e:
                print(f"Could not set timer resolution: {e}")

        last_report_time = time.time()
        loop_count = 0
        rate_report_interval = 1.0  # Report actual rate every second
        callback_decimation = 10  # Call callback every N iterations (~100Hz for UI updates)

        # Moving average for message period estimation
        message_period_history = []

        while not self._stop_flag.is_set():
            current_time = time.time()  # Single time call per iteration
            loop_start = current_time

            try:
                # Run the actuator communication
                self.actuator.run()

                # Read current state from motor (using stream data for efficiency)
                stream_data = self.actuator.get_stream_data()

                # Watchdog: Check if communication has stalled (no successful response in 2 seconds)
                time_since_last_response_us = self.actuator.time_since_last_response_microseconds()
                COMM_TIMEOUT_THRESHOLD_US = 2_000_000  # 2 seconds in microseconds
                if time_since_last_response_us > COMM_TIMEOUT_THRESHOLD_US:
                    raise MotorCommunicationError(
                        f"Communication timeout: no response for {time_since_last_response_us / 1_000_000:.1f}s"
                    )

                # Check for any motor errors and trigger reconnection
                if stream_data.errors != 0:
                    print(f"Motor error detected (error code: 0x{stream_data.errors:04X})")
                    raise MotorCommunicationError(f"Motor error 0x{stream_data.errors:04X}")

                # Calculate time since last measurement
                if self.last_measurement_time is not None:
                    dt = current_time - self.last_measurement_time
                    # Update estimated message period (moving average)
                    message_period_history.append(dt)
                    if len(message_period_history) > 100:
                        message_period_history.pop(0)
                    self.estimated_message_period_s = np.mean(message_period_history)
                else:
                    dt = 0.001  # Initial guess
                self.last_measurement_time = current_time

                # STATE ESTIMATOR: Predict step (fill in the blanks since last measurement)
                self.state_estimator.predict(self.last_force_command_N, dt)

                # STATE ESTIMATOR: Update step (correct prediction with measurement)
                position_measured_mm = stream_data.position / 1000.0
                self.state_estimator.update(position_measured_mm)

                # Get improved state estimates
                position_est_mm = self.state_estimator.get_position_mm()
                velocity_est_mm_s = self.state_estimator.get_velocity_mm_s()
                pos_uncertainty, vel_uncertainty = self.state_estimator.get_uncertainty()

                # Get current command (read-only, fast)
                control_mode = self.command.control_mode
                force_setpoint = self.command.force_setpoint_mN
                position_setpoint = self.command.position_setpoint_um

                # Execute control logic based on mode
                if control_mode == ControlMode.SLEEP:
                    # Sleep mode - no force output
                    if self.state.mode != MotorMode.SleepMode:
                        self.actuator.set_mode(MotorMode.SleepMode)
                        with self._state_lock:
                            self.state.mode = MotorMode.SleepMode

                elif control_mode == ControlMode.FORCE_DIRECT:
                    # Direct force control
                    if self.state.mode != MotorMode.ForceMode:
                        self.actuator.set_mode(MotorMode.ForceMode)
                        with self._state_lock:
                            self.state.mode = MotorMode.ForceMode

                    # Set the commanded force
                    self.actuator.set_streamed_force_mN(int(force_setpoint))

                elif control_mode == ControlMode.POSITION:
                    # Position control using PID
                    if self.state.mode != MotorMode.ForceMode:
                        self.actuator.set_mode(MotorMode.ForceMode)
                        with self._state_lock:
                            self.state.mode = MotorMode.ForceMode

                    # Compute PID output (reuse current_time)
                    force_command = self.pid.compute(
                        position_setpoint,
                        stream_data.position,
                        current_time
                    )

                    # Send force command to motor
                    self.actuator.set_streamed_force_mN(int(force_command))

                elif control_mode == ControlMode.SHOCK_PROFILE:
                    # Shock profile state machine with state estimation
                    if self.state.mode != MotorMode.ForceMode:
                        self.actuator.set_mode(MotorMode.ForceMode)
                        with self._state_lock:
                            self.state.mode = MotorMode.ForceMode

                    # Execute shock state machine using ESTIMATED states
                    with self._shock_lock:
                        current_shock_state = self.shock_state

                        if current_shock_state == ShockState.IDLE:
                            # Waiting - apply no force
                            self.actuator.set_streamed_force_mN(0)
                            self.last_force_command_N = 0.0

                        elif current_shock_state == ShockState.ACCELERATE:
                            # Mass = anvil + striker (both moving together)
                            self.state_estimator.set_mass(StateEstimator.MASS_TOTAL_KG)

                            # Accelerate force (negative direction)
                            accel_force_N = -self.shock_params.accel_force_N
                            accel_force_mN = accel_force_N * 1000.0
                            self.actuator.set_streamed_force_mN(int(accel_force_mN))
                            self.last_force_command_N = accel_force_N

                            # LOOK-AHEAD: Predict where we'll be in next few messages
                            time_ahead = self.look_ahead_messages * self.estimated_message_period_s
                            pos_future, vel_future = self.state_estimator.predict_future_state(
                                accel_force_N, time_ahead
                            )

                            # Check if we WILL reach switch position (use future prediction!)
                            if pos_future < -self.shock_params.switch_position_mm:
                                self.shock_state = ShockState.DECELERATE
                                self.shock_first_crossing = False
                                # Update mass for deceleration (striker decouples!)
                                self.state_estimator.set_mass(StateEstimator.MASS_ANVIL_KG)
                                print(f"Shock: ACCELERATE → DECELERATE at est={position_est_mm:.2f}mm, "
                                      f"predicted={pos_future:.2f}mm, "
                                      f"measured={position_measured_mm:.2f}mm, "
                                      f"vel={velocity_est_mm_s:.0f}mm/s")

                        elif current_shock_state == ShockState.DECELERATE:
                            # Mass = anvil only (striker decoupled, in free flight)
                            # Already set during transition, but keep it updated
                            self.state_estimator.set_mass(StateEstimator.MASS_ANVIL_KG)

                            # Decelerate force (negative direction)
                            decel_force_N = -self.shock_params.decel_force_N
                            decel_force_mN = decel_force_N * 1000.0
                            self.actuator.set_streamed_force_mN(int(decel_force_mN))
                            self.last_force_command_N = decel_force_N

                            # Track crossings using ESTIMATED position
                            if not self.shock_first_crossing:
                                # Waiting for first crossing (forward in negative direction)
                                if position_est_mm < -self.shock_params.end_position_mm:
                                    self.shock_first_crossing = True
                                    print(f"Shock: First crossing at est={position_est_mm:.2f}mm, "
                                          f"vel={velocity_est_mm_s:.0f}mm/s (forward)")
                            else:
                                # Waiting for second crossing (backward toward zero)
                                if position_est_mm > -self.shock_params.end_position_mm:
                                    self.shock_state = ShockState.STABILIZE
                                    self.shock_stabilize_position_um = stream_data.position
                                    # Increase process noise after collision (model breaks down)
                                    self.state_estimator.increase_process_noise(factor=20.0)
                                    print(f"Shock: Second crossing at est={position_est_mm:.2f}mm, "
                                          f"vel={velocity_est_mm_s:.0f}mm/s (backward) → STABILIZE")

                        elif current_shock_state == ShockState.STABILIZE:
                            # Switch to position control at stabilize position
                            force_command = self.pid.compute(
                                self.shock_stabilize_position_um,
                                stream_data.position,
                                current_time
                            )
                            self.actuator.set_streamed_force_mN(int(force_command))

                            # Check if we need to wait before homing
                            if self.shock_current_repetition < self.shock_params.repetitions - 1:
                                # More repetitions to go - transition to WAIT
                                self.shock_state = ShockState.WAIT
                                self.shock_wait_start_time = current_time
                                print(f"Shock: STABILIZE → WAIT (rep {self.shock_current_repetition + 1}/{self.shock_params.repetitions})")
                            else:
                                # All repetitions complete
                                self.shock_state = ShockState.IDLE
                                print(f"Shock profile complete: {self.shock_params.repetitions} repetition(s)")

                        elif current_shock_state == ShockState.WAIT:
                            # Hold position while waiting
                            force_command = self.pid.compute(
                                self.shock_stabilize_position_um,
                                stream_data.position,
                                current_time
                            )
                            self.actuator.set_streamed_force_mN(int(force_command))

                            # Thermal management: pause if too hot, resume when cool
                            current_temp = stream_data.temperature
                            if current_temp >= 110:
                                if not self.shock_thermal_pause:
                                    self.shock_thermal_pause = True
                                    print(f"Shock: THERMAL PAUSE activated at {current_temp}°C (waiting for cool-down to 45°C)")
                            elif current_temp <= 45:
                                if self.shock_thermal_pause:
                                    self.shock_thermal_pause = False
                                    print(f"Shock: THERMAL PAUSE cleared at {current_temp}°C (resuming operations)")

                            # Check if wait time elapsed and not in thermal pause
                            wait_time_elapsed = current_time - self.shock_wait_start_time >= self.shock_params.wait_time_s
                            if wait_time_elapsed and not self.shock_thermal_pause:
                                self.shock_state = ShockState.HOMING
                                print(f"Shock: WAIT → HOMING")

                        elif current_shock_state == ShockState.HOMING:
                            # Reset process noise for homing (predictable motion)
                            self.state_estimator.reset_process_noise()

                            # Apply homing force to return toward zero
                            homing_force_N = self.shock_params.homing_force_N
                            homing_force_mN = homing_force_N * 1000.0
                            self.actuator.set_streamed_force_mN(int(homing_force_mN))
                            self.last_force_command_N = homing_force_N

                            # Check if back at home using ESTIMATED position
                            if position_est_mm > -30.0:
                                # Back at home - start next repetition
                                self.shock_current_repetition += 1
                                self.shock_state = ShockState.ACCELERATE
                                self.shock_first_crossing = False
                                print(f"Shock: HOMING → ACCELERATE at est={position_est_mm:.2f}mm "
                                      f"(starting rep {self.shock_current_repetition + 1}/{self.shock_params.repetitions})")

                # Update state (fast, no copy)
                with self._state_lock:
                    self.state.position_um = stream_data.position
                    self.state.force_mN = stream_data.force
                    self.state.power_W = stream_data.power
                    self.state.temperature_C = stream_data.temperature
                    self.state.voltage_mV = stream_data.voltage
                    self.state.errors = stream_data.errors
                    self.state.control_mode = control_mode
                    self.state.shock_state = self.shock_state
                    self.state.thermal_pause = self.shock_thermal_pause

                # Call callback only every N iterations (decimation for UI updates)
                loop_count += 1
                if self.state_update_callback and (loop_count % callback_decimation == 0):
                    # Create state copy only when needed for callback
                    state_copy = MotorState(
                        position_um=stream_data.position,
                        force_mN=stream_data.force,
                        power_W=stream_data.power,
                        temperature_C=stream_data.temperature,
                        voltage_mV=stream_data.voltage,
                        errors=stream_data.errors,
                        mode=self.state.mode,
                        control_mode=control_mode,
                        connected=self.state.connected,
                        running=self.state.running,
                        shock_state=self.shock_state,
                        thermal_pause=self.shock_thermal_pause
                    )
                    self.state_update_callback(state_copy)

            except (MotorCommunicationError, OrcaError) as e:
                # Motor communication errors - attempt automatic reconnection
                print(f"Communication error in control loop: {e}")
                print("Attempting automatic reconnection...")

                # Attempt reconnection (up to 3 tries)
                reconnect_success = False
                for attempt in range(3):
                    if self.reconnect():
                        reconnect_success = True
                        print("Auto-reconnect successful, resuming control loop")
                        break
                    else:
                        if attempt < 2:
                            print(f"Reconnection attempt {attempt + 1} failed, retrying in 1 second...")
                            time.sleep(1.0)

                if not reconnect_success:
                    print("Failed to reconnect after 3 attempts. Will retry on next cycle...")
                    time.sleep(0.5)

                # Continue loop (shock profile state is preserved)
                continue

            except Exception as e:
                # Other exceptions (code bugs) - log but don't reconnect
                print(f"ERROR in control loop (not communication-related): {e}")
                import traceback
                traceback.print_exc()
                # Brief pause before continuing
                time.sleep(0.1)

            # Maintain update rate (if specified)
            if self.update_period > 0:
                elapsed = time.time() - loop_start
                sleep_time = self.update_period - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                elif sleep_time < -self.update_period * 0.1:  # More than 10% overrun
                    # Warn if loop is consistently too slow (only occasionally to avoid spam)
                    if loop_count % 100 == 0:
                        print(f"WARNING: Control loop running {abs(sleep_time)*1000:.1f}ms behind target ({abs(sleep_time/self.update_period)*100:.0f}% overrun)")

            # Report actual loop rate periodically (check every 1000 iterations to reduce overhead)
            if loop_count % 1000 == 0:
                current_time = time.time()
                if current_time - last_report_time >= rate_report_interval:
                    actual_rate = loop_count / (current_time - last_report_time)
                    print(f"Control loop rate: {actual_rate:.1f} Hz")
                    loop_count = 0
                    last_report_time = current_time

        # Cleanup timer resolution on Windows
        if platform.system() == 'Windows':
            try:
                import ctypes
                ctypes.windll.winmm.timeEndPeriod(1)
            except:
                pass

        print("Control loop stopped")
