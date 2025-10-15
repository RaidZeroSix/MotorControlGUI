"""
Motor Controller Module

Manages the Orca motor communication and control loop.
Uses Force Mode and Sleep Mode only, with custom PID position control.
"""

import threading
import time
import json
import os
from typing import Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

from pyorcasdk import Actuator, MotorMode, OrcaError
from pid_controller import PIDController, PIDParameters


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

        # Profiles directory
        self.profiles_dir = Path("profiles")
        self.profiles_dir.mkdir(exist_ok=True)

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

        self._stop_flag.clear()
        self._control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._control_thread.start()

        with self._state_lock:
            self.state.running = True

        return True

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
            print("Position zeroed")
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

        last_report_time = time.time()
        loop_count = 0
        rate_report_interval = 1.0  # Report actual rate every second
        callback_decimation = 10  # Call callback every N iterations (~100Hz for UI updates)

        while not self._stop_flag.is_set():
            current_time = time.time()  # Single time call per iteration
            loop_start = current_time

            try:
                # Run the actuator communication
                self.actuator.run()

                # Read current state from motor (using stream data for efficiency)
                stream_data = self.actuator.get_stream_data()

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
                    # Shock profile state machine
                    if self.state.mode != MotorMode.ForceMode:
                        self.actuator.set_mode(MotorMode.ForceMode)
                        with self._state_lock:
                            self.state.mode = MotorMode.ForceMode

                    # Get current position in mm
                    position_mm = stream_data.position / 1000.0

                    # Execute shock state machine
                    with self._shock_lock:
                        current_shock_state = self.shock_state

                        if current_shock_state == ShockState.IDLE:
                            # Waiting - apply no force
                            self.actuator.set_streamed_force_mN(0)

                        elif current_shock_state == ShockState.ACCELERATE:
                            # Accelerate until switch position (negative direction - flip force sign)
                            accel_force_mN = -self.shock_params.accel_force_N * 1000.0
                            self.actuator.set_streamed_force_mN(int(accel_force_mN))

                            # Check if reached switch position (going negative)
                            if position_mm < -self.shock_params.switch_position_mm:
                                self.shock_state = ShockState.DECELERATE
                                self.shock_first_crossing = False
                                print(f"Shock: ACCELERATE → DECELERATE at {position_mm:.2f}mm")

                        elif current_shock_state == ShockState.DECELERATE:
                            # Decelerate/reverse - track crossings (flip force sign)
                            decel_force_mN = -self.shock_params.decel_force_N * 1000.0
                            self.actuator.set_streamed_force_mN(int(decel_force_mN))

                            # Track crossings of end_position (negative direction)
                            if not self.shock_first_crossing:
                                # Waiting for first crossing (forward in negative direction)
                                if position_mm < -self.shock_params.end_position_mm:
                                    self.shock_first_crossing = True
                                    print(f"Shock: First crossing at {position_mm:.2f}mm (forward)")
                            else:
                                # Waiting for second crossing (backward toward zero)
                                if position_mm > -self.shock_params.end_position_mm:
                                    self.shock_state = ShockState.STABILIZE
                                    self.shock_stabilize_position_um = stream_data.position
                                    print(f"Shock: Second crossing at {position_mm:.2f}mm (backward) → STABILIZE")

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

                            # Check if wait time elapsed
                            if current_time - self.shock_wait_start_time >= self.shock_params.wait_time_s:
                                self.shock_state = ShockState.HOMING
                                print(f"Shock: WAIT → HOMING")

                        elif current_shock_state == ShockState.HOMING:
                            # Apply homing force to return to 5mm (not zero, for tolerance)
                            homing_force_mN = self.shock_params.homing_force_N * 1000.0
                            self.actuator.set_streamed_force_mN(int(homing_force_mN))

                            # Check if back at home position (5mm ± 2mm)
                            target_home_mm = 5.0
                            position_tolerance_mm = 2.0
                            if abs(position_mm - target_home_mm) < position_tolerance_mm:
                                # Back at home - start next repetition
                                self.shock_current_repetition += 1
                                self.shock_state = ShockState.ACCELERATE
                                self.shock_first_crossing = False
                                print(f"Shock: HOMING → ACCELERATE (starting rep {self.shock_current_repetition + 1}/{self.shock_params.repetitions})")

                # Update state (fast, no copy)
                with self._state_lock:
                    self.state.position_um = stream_data.position
                    self.state.force_mN = stream_data.force
                    self.state.power_W = stream_data.power
                    self.state.temperature_C = stream_data.temperature
                    self.state.voltage_mV = stream_data.voltage
                    self.state.errors = stream_data.errors
                    self.state.control_mode = control_mode

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
                        running=self.state.running
                    )
                    self.state_update_callback(state_copy)

            except Exception as e:
                print(f"Error in control loop: {e}")

            # Maintain update rate (if specified)
            if self.update_period > 0:
                elapsed = time.time() - loop_start
                sleep_time = self.update_period - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            # Report actual loop rate periodically (check every 1000 iterations to reduce overhead)
            if loop_count % 1000 == 0:
                current_time = time.time()
                if current_time - last_report_time >= rate_report_interval:
                    actual_rate = loop_count / (current_time - last_report_time)
                    print(f"Control loop rate: {actual_rate:.1f} Hz")
                    loop_count = 0
                    last_report_time = current_time

        print("Control loop stopped")
