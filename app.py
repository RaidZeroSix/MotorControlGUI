"""
NiceGUI Application for Orca Motor Control

Provides a web-based interface for controlling the Orca motor with
Force Mode and Sleep Mode, including custom PID position control.
"""

from nicegui import ui, app
from motor_controller import MotorController, ControlMode, MotorState
import asyncio
from typing import Optional, List
from collections import deque
import time
import serial.tools.list_ports


def get_available_ports() -> List[tuple]:
    """
    Scan for available serial ports.

    Returns:
        List of tuples (port_device, port_description)
        e.g., [('/dev/ttyUSB0', 'USB Serial Port'), ('COM3', 'Communications Port')]
    """
    ports = serial.tools.list_ports.comports()
    available_ports = []

    for port in ports:
        # Create a user-friendly description
        if port.description and port.description != 'n/a':
            description = f"{port.device} - {port.description}"
        else:
            description = port.device

        available_ports.append((port.device, description))

    return available_ports


class MotorGUI:
    """Main GUI application class"""

    def __init__(self):
        # Initialize controller at maximum speed (0 = no sleep between iterations)
        # Run as fast as hardware allows for best Modbus communication rate
        self.controller = MotorController(update_rate_hz=0)

        # UI state
        self.connected = False
        self.e_stop_active = False

        # Data for plotting (5 seconds at ~100Hz = 500 points)
        self.plot_window_seconds = 5.0
        self.max_plot_points = 500
        self.time_data = deque(maxlen=self.max_plot_points)
        self.position_data = deque(maxlen=self.max_plot_points)
        self.force_data = deque(maxlen=self.max_plot_points)
        self.start_time = time.time()

        # UI elements (will be created in create_ui)
        self.status_label: Optional[ui.label] = None
        self.position_label: Optional[ui.label] = None
        self.force_label: Optional[ui.label] = None
        self.telemetry_labels: dict = {}
        self.position_chart = None
        self.force_chart = None
        self.connect_btn = None
        self.disconnect_btn = None
        self.e_stop_btn = None
        self.shock_state_label: Optional[ui.label] = None
        self.thermal_pause_label: Optional[ui.label] = None
        self.profile_select: Optional[ui.label] = None

        # Mode switching
        self.mode_toggle: Optional[ui.toggle] = None
        self.operator_card: Optional[ui.card] = None
        self.developer_card: Optional[ui.card] = None

        # Operator mode UI elements
        self.operator_profile_select: Optional[ui.select] = None
        self.operator_repetitions_input: Optional[ui.number] = None
        self.operator_homing_force_input: Optional[ui.number] = None
        self.operator_shock_state_label: Optional[ui.label] = None
        self.operator_thermal_pause_label: Optional[ui.label] = None

        # Register callback for motor state updates
        self.controller.state_update_callback = self._on_state_update

    def _on_state_update(self, state: MotorState):
        """Callback for motor state updates from control thread"""
        # Add data to plot queues
        current_time = time.time() - self.start_time
        self.time_data.append(current_time)
        self.position_data.append(state.position_um / 1000.0)  # Convert to mm
        self.force_data.append(state.force_mN / 1000.0)  # Convert to N

    async def _update_ui(self):
        """Periodic UI update"""
        state = self.controller.get_state()

        # Enforce sleep mode when E-STOP is active
        if self.e_stop_active and state.control_mode != ControlMode.SLEEP:
            self.controller.set_control_mode(ControlMode.SLEEP)

        # Update status
        if self.status_label:
            if state.connected:
                if state.running:
                    mode_str = state.control_mode.value.upper()
                    self.status_label.text = f"Status: Connected - {mode_str}"
                    self.status_label.classes('text-green-600', remove='text-red-600 text-yellow-600')
                else:
                    self.status_label.text = "Status: Connected (Stopped)"
                    self.status_label.classes('text-yellow-600', remove='text-red-600 text-green-600')
            else:
                self.status_label.text = "Status: Disconnected"
                self.status_label.classes('text-red-600', remove='text-green-600 text-yellow-600')

        # Update position and force
        if self.position_label:
            self.position_label.text = f"Position: {state.position_um / 1000.0:.2f} mm"
        if self.force_label:
            self.force_label.text = f"Force: {state.force_mN / 1000.0:.2f} N"

        # Update telemetry
        if self.telemetry_labels:
            self.telemetry_labels['power'].text = f"{state.power_W} W"
            self.telemetry_labels['temp'].text = f"{state.temperature_C} °C"
            self.telemetry_labels['voltage'].text = f"{state.voltage_mV / 1000.0:.1f} V"
            self.telemetry_labels['errors'].text = f"0x{state.errors:04X}"

        # Update shock profile state (developer mode)
        if self.shock_state_label:
            self.shock_state_label.text = f"State: {state.shock_state.value.upper()}"

        # Update thermal pause indicator (developer mode)
        if self.thermal_pause_label:
            self.thermal_pause_label.visible = state.thermal_pause

        # Update shock profile state (operator mode)
        if self.operator_shock_state_label:
            self.operator_shock_state_label.text = f"State: {state.shock_state.value.upper()}"

        # Update thermal pause indicator (operator mode)
        if self.operator_thermal_pause_label:
            self.operator_thermal_pause_label.visible = state.thermal_pause

    def create_ui(self):
        """Create the main UI"""
        ui.page_title('Pixels On Target')

        with ui.header().classes('items-center justify-between'):
            ui.label('Pixels On Target').classes('text-h4')
            self.status_label = ui.label('Status: Disconnected').classes('text-red-600')

        # Mode toggle
        with ui.card().classes('w-full'):
            with ui.row().classes('items-center gap-4'):
                ui.label('Interface Mode:').classes('font-bold')
                self.mode_toggle = ui.toggle(['Operator', 'Developer'], value='Operator', on_change=self._on_mode_toggle)

        # Top row - Connection and control
        with ui.card().classes('w-full'):
            ui.label('Connection & Control').classes('text-h6')

            with ui.row().classes('w-full items-end gap-4'):
                # Scan for available ports
                available_ports = get_available_ports()
                if available_ports:
                    port_options = {device: desc for device, desc in available_ports}
                    default_port = available_ports[0][0]
                else:
                    port_options = {'': 'No ports found'}
                    default_port = ''

                port_select = ui.select(
                    options=port_options,
                    value=default_port,
                    label='Serial Port'
                ).style('min-width: 300px')

                ui.button('', icon='refresh', on_click=lambda: self._refresh_ports(port_select)).props('flat')

                baud_input = ui.number('Baud Rate', value=1000000, format='%d').style('width: 120px')

                # Connect/Disconnect buttons (mutually exclusive)
                connect_btn = ui.button('Connect', on_click=lambda: self._connect(
                    port_select.value, int(baud_input.value)
                )).props('color=green')

                disconnect_btn = ui.button('Disconnect', on_click=self._disconnect).props('color=red')
                disconnect_btn.visible = False  # Hidden until connected

                ui.separator().props('vertical')

                ui.button('Zero Position', on_click=self._zero_position).props('color=primary flat')
                e_stop_btn = ui.button('E-STOP', on_click=self._emergency_stop).props('color=red')

                # Store button references for toggling visibility
                self.connect_btn = connect_btn
                self.disconnect_btn = disconnect_btn
                self.e_stop_btn = e_stop_btn

        # Second row - Telemetry
        with ui.card().classes('w-full'):
            ui.label('Telemetry').classes('text-h6')
            with ui.row().classes('w-full items-center gap-8'):
                with ui.row().classes('items-center gap-2'):
                    ui.label('Position:').classes('font-bold')
                    self.position_label = ui.label('-- mm').classes('text-lg')
                with ui.row().classes('items-center gap-2'):
                    ui.label('Force:').classes('font-bold')
                    self.force_label = ui.label('-- N').classes('text-lg')
                with ui.row().classes('items-center gap-2'):
                    ui.label('Power:').classes('font-bold')
                    self.telemetry_labels['power'] = ui.label('-- W')
                with ui.row().classes('items-center gap-2'):
                    ui.label('Temp:').classes('font-bold')
                    self.telemetry_labels['temp'] = ui.label('-- °C')
                with ui.row().classes('items-center gap-2'):
                    ui.label('Voltage:').classes('font-bold')
                    self.telemetry_labels['voltage'] = ui.label('-- V')
                with ui.row().classes('items-center gap-2'):
                    ui.label('Errors:').classes('font-bold')
                    self.telemetry_labels['errors'] = ui.label('0x0000')

        # Operator Mode Interface
        with ui.card().classes('w-full') as operator_card:
            self._create_operator_interface()
        self.operator_card = operator_card
        operator_card.visible = True  # Default to operator mode

        # Developer Mode Interface (existing Control setpoints)
        with ui.card().classes('w-full') as developer_card:
            ui.label('Control Setpoints').classes('text-h6')
            with ui.row().classes('w-full items-center gap-4').style('min-height: 60px'):
                # Mode selection
                mode_select = ui.select(
                    ['Sleep', 'Force Direct', 'Position', 'Shock Profile'],
                    value='Sleep',
                    label='Mode'
                ).style('width: 150px')

                # Force control (conditionally visible)
                with ui.row().classes('items-center gap-2') as force_row:
                    force_input = ui.number('Force (N)', value=0.0, step=0.1, format='%.2f').style('width: 110px')
                    ui.button('Apply', on_click=lambda: self._set_force(force_input.value)).props('flat dense')
                force_row.visible = False  # Hidden by default (Sleep mode)

                # Position control (conditionally visible)
                with ui.row().classes('items-center gap-2') as position_row:
                    position_input = ui.number('Position (mm)', value=35.0, step=0.1, format='%.2f').style('width: 110px')
                    ui.button('Apply', on_click=lambda: self._set_position(position_input.value)).props('flat dense')
                position_row.visible = False  # Hidden by default (Sleep mode)

                # PID tuning (conditionally visible - only for Position mode)
                with ui.row().classes('items-center gap-2') as pid_row:
                    ui.label('PID:').classes('text-sm')
                    kp_input = ui.number('Kp', value=0.1, step=0.01, format='%.3f').style('width: 90px')
                    ki_input = ui.number('Ki', value=0.01, step=0.001, format='%.4f').style('width: 90px')
                    kd_input = ui.number('Kd', value=0.005, step=0.001, format='%.4f').style('width: 90px')

                    ui.separator().props('vertical')

                    ui.label('Limits:').classes('text-sm')
                    max_force_input = ui.number('Max (N)', value=30, step=10, format='%.0f').style('width: 90px')
                    min_force_input = ui.number('Min (N)', value=-30, step=10, format='%.0f').style('width: 90px')

                    ui.button('Update', on_click=lambda: self._update_pid_and_limits(
                        kp_input.value, ki_input.value, kd_input.value,
                        max_force_input.value, min_force_input.value
                    )).props('flat dense')
                pid_row.visible = False  # Hidden by default (Sleep mode)

                # Shock profile control (conditionally visible) - use column layout
                with ui.column().classes('w-full gap-2') as shock_container:
                    # Row 1: Profile management
                    with ui.row().classes('items-center gap-2'):
                        profile_name_input = ui.input('Profile Name', value='Default Profile').style('width: 200px')

                        # Profile selector dropdown (populated dynamically)
                        profile_select = ui.select(
                            options=[],
                            label='Load Profile',
                            on_change=lambda e: self._load_profile(e.value) if e.value else None
                        ).style('width: 200px')

                        ui.button('Save', on_click=lambda: self._save_profile(profile_name_input.value)).props('color=primary flat dense')
                        ui.button('Refresh', on_click=lambda: self._refresh_profiles(profile_select)).props('flat dense icon=refresh')

                    # Row 2: Motion parameters
                    with ui.row().classes('items-center gap-2'):
                        accel_force_input = ui.number('Accel Force (N)', value=150.0, step=10, format='%.0f').style('width: 120px')
                        decel_force_input = ui.number('Decel Force (N)', value=-180.0, step=10, format='%.0f').style('width: 120px')
                        switch_pos_input = ui.number('Switch (mm)', value=60.0, step=1, format='%.1f').style('width: 100px')
                        end_pos_input = ui.number('End (mm)', value=100.0, step=1, format='%.1f').style('width: 100px')

                    # Row 3: Repetition parameters and controls
                    with ui.row().classes('items-center gap-2'):
                        repetitions_input = ui.number('Repetitions', value=1, step=1, format='%d', min=1).style('width: 100px')
                        wait_time_input = ui.number('Wait Time (s)', value=2.0, step=0.5, format='%.1f').style('width: 110px')
                        homing_force_input = ui.number('Homing Force (N)', value=50.0, step=10, format='%.0f').style('width: 130px')

                        ui.separator().props('vertical')

                        shock_state_label = ui.label('State: IDLE').classes('text-sm font-bold')

                        thermal_pause_label = ui.label('🔥 THERMAL PAUSE').classes('text-sm font-bold text-orange-600')
                        thermal_pause_label.visible = False

                        ui.button('Start', on_click=lambda: self._start_shock_profile(
                            profile_name_input.value,
                            accel_force_input.value, decel_force_input.value,
                            switch_pos_input.value, end_pos_input.value,
                            int(repetitions_input.value), wait_time_input.value, homing_force_input.value
                        )).props('color=green flat dense')

                        ui.button('Abort', on_click=self._abort_shock_profile).props('color=red flat dense')

                shock_container.visible = False  # Hidden by default (Sleep mode)

                # Store references
                self.profile_select = profile_select

                # Store reference to shock state label for updates
                self.shock_state_label = shock_state_label
                self.thermal_pause_label = thermal_pause_label

                # Update visibility when mode changes
                def on_mode_change(e):
                    mode = e.value
                    force_row.visible = (mode == 'Force Direct')
                    position_row.visible = (mode == 'Position')
                    pid_row.visible = (mode == 'Position')
                    shock_container.visible = (mode == 'Shock Profile')
                    if mode == 'Shock Profile':
                        self._refresh_profiles(profile_select)
                    self._set_mode(mode)

                mode_select.on_value_change(on_mode_change)

        self.developer_card = developer_card
        developer_card.visible = False  # Hidden by default (Operator mode is default)

        # Start periodic UI update timer
        ui.timer(0.1, self._update_ui)

    def _create_operator_interface(self):
        """Create the simplified operator mode interface"""
        ui.label('Operator Mode').classes('text-h6')

        # Homing Panel
        with ui.card().classes('w-full'):
            ui.label('Homing Panel').classes('text-subtitle1 font-bold')
            with ui.row().classes('items-center gap-4'):
                self.operator_homing_force_input = ui.number(
                    'Homing Force (N)',
                    value=50.0,
                    step=10,
                    format='%.0f'
                ).style('width: 130px')
                ui.button('Home', on_click=self._run_homing).props('color=primary')

        # Shock Panel
        with ui.card().classes('w-full'):
            ui.label('Shock Panel').classes('text-subtitle1 font-bold')

            with ui.row().classes('items-center gap-4'):
                # Profile selector
                self.operator_profile_select = ui.select(
                    options=[],
                    label='Profile',
                    on_change=lambda e: self._load_profile_operator(e.value) if e.value else None
                ).style('width: 250px')

                # Repetitions
                self.operator_repetitions_input = ui.number(
                    'Rounds',
                    value=1,
                    step=1,
                    format='%d',
                    min=1
                ).style('width: 100px')

                ui.separator().props('vertical')

                # State indicator
                self.operator_shock_state_label = ui.label('State: IDLE').classes('text-sm font-bold')

                # Thermal pause indicator
                self.operator_thermal_pause_label = ui.label('🔥 THERMAL PAUSE').classes('text-sm font-bold text-orange-600')
                self.operator_thermal_pause_label.visible = False

                ui.separator().props('vertical')

                # Start/Abort buttons
                ui.button('Start', on_click=self._start_shock_operator).props('color=green')
                ui.button('Abort', on_click=self._abort_shock_profile).props('color=red')

        # Load profiles initially
        self._refresh_profiles_operator()

    def _on_mode_toggle(self, e):
        """Handle mode toggle between Operator and Developer"""
        if self.operator_card and self.developer_card:
            is_operator = (e.value == 'Operator')
            self.operator_card.visible = is_operator
            self.developer_card.visible = not is_operator

            # Put motor to sleep when switching modes for safety
            if self.connected:
                self.controller.set_control_mode(ControlMode.SLEEP)
                ui.notify('Motor set to Sleep mode', type='info')

    def _refresh_ports(self, port_select):
        """Refresh the list of available serial ports"""
        available_ports = get_available_ports()

        if available_ports:
            # Update options
            port_options = {device: desc for device, desc in available_ports}
            port_select.options = port_options
            port_select.value = available_ports[0][0]
            ui.notify(f'Found {len(available_ports)} port(s)', type='info')
        else:
            port_select.options = {'': 'No ports found'}
            port_select.value = ''
            ui.notify('No serial ports found', type='warning')

        port_select.update()

    def _connect(self, port: str, baud_rate: int):
        """Connect to motor and start control loop"""
        if not port:
            ui.notify('Please select a serial port', type='warning')
            return

        if self.controller.connect(port, baud_rate):
            ui.notify(f'Connected to {port}', type='positive')
            self.connected = True

            # Toggle button visibility
            if self.connect_btn:
                self.connect_btn.visible = False
            if self.disconnect_btn:
                self.disconnect_btn.visible = True

            # Auto-start control loop
            if self.controller.start_control_loop():
                ui.notify('Control loop started', type='info')
            else:
                ui.notify('Failed to start control loop', type='negative')
        else:
            ui.notify('Connection failed', type='negative')

    def _disconnect(self):
        """Stop control loop and disconnect from motor"""
        # Stop control loop first
        self.controller.stop_control_loop()

        # Then disconnect
        self.controller.disconnect()
        ui.notify('Disconnected', type='info')
        self.connected = False

        # Toggle button visibility
        if self.connect_btn:
            self.connect_btn.visible = True
        if self.disconnect_btn:
            self.disconnect_btn.visible = False

    def _emergency_stop(self):
        """Toggle emergency stop"""
        self.e_stop_active = not self.e_stop_active

        if self.e_stop_active:
            # E-STOP activated - put motor to sleep
            self.controller.set_control_mode(ControlMode.SLEEP)
            if self.e_stop_btn:
                self.e_stop_btn.props('color=orange')
                self.e_stop_btn.text = 'E-STOP ACTIVE'
            ui.notify('E-STOP ACTIVATED - All operations locked', type='negative')
        else:
            # E-STOP deactivated
            if self.e_stop_btn:
                self.e_stop_btn.props('color=red')
                self.e_stop_btn.text = 'E-STOP'
            ui.notify('E-STOP deactivated', type='positive')

    def _zero_position(self):
        """Zero the position reference"""
        self.controller.zero_position()
        # Set motor to sleep mode for safety (clears any active forces)
        self.controller.set_control_mode(ControlMode.SLEEP)
        ui.notify('Position zeroed and motor set to Sleep mode', type='info')

    def _set_mode(self, mode_str: str):
        """Set control mode"""
        if self.e_stop_active and mode_str != 'Sleep':
            ui.notify('E-STOP is active - Cannot change mode', type='warning')
            return

        mode_map = {
            'Sleep': ControlMode.SLEEP,
            'Force Direct': ControlMode.FORCE_DIRECT,
            'Position': ControlMode.POSITION,
            'Shock Profile': ControlMode.SHOCK_PROFILE
        }
        mode = mode_map.get(mode_str, ControlMode.SLEEP)
        self.controller.set_control_mode(mode)
        ui.notify(f'Mode set to {mode_str}', type='info')

    def _set_force(self, force_n: float):
        """Set force setpoint"""
        if self.e_stop_active:
            ui.notify('E-STOP is active - Cannot set force', type='warning')
            return

        force_mN = force_n * 1000.0  # Convert N to mN
        self.controller.set_force_setpoint(force_mN)
        ui.notify(f'Force setpoint: {force_n:.2f} N', type='info')

    def _set_position(self, position_mm: float):
        """Set position setpoint"""
        if self.e_stop_active:
            ui.notify('E-STOP is active - Cannot set position', type='warning')
            return

        position_um = position_mm * 1000.0  # Convert mm to um
        self.controller.set_position_setpoint(position_um)
        ui.notify(f'Position setpoint: {position_mm:.2f} mm', type='info')

    def _update_pid(self, kp: float, ki: float, kd: float):
        """Update PID parameters"""
        if self.e_stop_active:
            ui.notify('E-STOP is active - Cannot update PID', type='warning')
            return

        self.controller.update_pid_parameters(kp=kp, ki=ki, kd=kd)
        ui.notify(f'PID updated: Kp={kp:.3f}, Ki={ki:.4f}, Kd={kd:.4f}', type='info')

    def _update_pid_and_limits(self, kp: float, ki: float, kd: float, max_force_n: float, min_force_n: float):
        """Update PID parameters and force limits"""
        if self.e_stop_active:
            ui.notify('E-STOP is active - Cannot update PID', type='warning')
            return

        max_force_mN = max_force_n * 1000.0
        min_force_mN = min_force_n * 1000.0
        self.controller.update_pid_parameters(kp=kp, ki=ki, kd=kd, max_output=max_force_mN, min_output=min_force_mN)
        ui.notify(f'PID updated: Kp={kp:.3f}, Ki={ki:.4f}, Kd={kd:.4f} | Limits: [{min_force_n:.0f}, {max_force_n:.0f}] N', type='info')

    def _start_shock_profile(self, name: str, accel_force_n: float, decel_force_n: float,
                            switch_pos_mm: float, end_pos_mm: float,
                            repetitions: int, wait_time_s: float, homing_force_n: float):
        """Start shock profile execution with given parameters"""
        if self.e_stop_active:
            ui.notify('E-STOP is active - Cannot start shock profile', type='warning')
            return

        # Update parameters
        self.controller.set_shock_profile_parameters(
            name=name,
            accel_force_N=accel_force_n,
            decel_force_N=decel_force_n,
            switch_position_mm=switch_pos_mm,
            end_position_mm=end_pos_mm,
            repetitions=repetitions,
            wait_time_s=wait_time_s,
            homing_force_N=homing_force_n
        )
        # Start execution
        self.controller.start_shock_profile()
        ui.notify(f'Shock profile started: {name} ({repetitions} rep)', type='positive')

    def _abort_shock_profile(self):
        """Abort shock profile execution"""
        self.controller.abort_shock_profile()
        ui.notify('Shock profile aborted', type='warning')

    def _save_profile(self, name: str):
        """Save current shock profile to disk"""
        if not name or name.strip() == '':
            ui.notify('Please enter a profile name', type='warning')
            return

        if self.controller.save_shock_profile(name):
            ui.notify(f'Profile saved: {name}', type='positive')
            if self.profile_select:
                self._refresh_profiles(self.profile_select)
        else:
            ui.notify('Failed to save profile', type='negative')

    def _load_profile(self, name: str):
        """Load shock profile from disk and update GUI"""
        if not name:
            return

        if self.controller.load_shock_profile(name):
            ui.notify(f'Profile loaded: {name}', type='positive')
            # TODO: Update GUI inputs with loaded parameters
            # This would require storing references to all input widgets
        else:
            ui.notify('Failed to load profile', type='negative')

    def _refresh_profiles(self, profile_select):
        """Refresh the list of available profiles"""
        profiles = self.controller.list_shock_profiles()
        if profiles:
            profile_select.options = [''] + profiles  # Empty option at start
            profile_select.value = ''
        else:
            profile_select.options = ['']
            profile_select.value = ''
        profile_select.update()

    def _run_homing(self):
        """Run homing operation with specified force"""
        if not self.connected:
            ui.notify('Please connect to motor first', type='warning')
            return

        if self.e_stop_active:
            ui.notify('E-STOP is active - Cannot run homing', type='warning')
            return

        homing_force = self.operator_homing_force_input.value if self.operator_homing_force_input else 50.0

        # Set to Force Direct mode and apply homing force
        self.controller.set_control_mode(ControlMode.FORCE_DIRECT)
        self.controller.set_force_setpoint(homing_force * 1000)  # Convert to mN
        ui.notify(f'Homing with {homing_force}N force', type='info')

    def _start_shock_operator(self):
        """Start shock profile from operator mode"""
        if not self.connected:
            ui.notify('Please connect to motor first', type='warning')
            return

        if self.e_stop_active:
            ui.notify('E-STOP is active - Cannot start shock profile', type='warning')
            return

        # Load the selected profile (or use default)
        profile_name = self.operator_profile_select.value if self.operator_profile_select else '1000 G Shock - SCAR 308'

        # Load profile to get parameters
        if self.controller.load_shock_profile(profile_name):
            # Update repetitions from operator input
            params = self.controller.shock_params
            repetitions = int(self.operator_repetitions_input.value) if self.operator_repetitions_input else 1

            # Start with updated repetitions
            self.controller.set_shock_profile_parameters(
                name=params.name,
                accel_force_N=params.accel_force_N,
                decel_force_N=params.decel_force_N,
                switch_position_mm=params.switch_position_mm,
                end_position_mm=params.end_position_mm,
                repetitions=repetitions,
                wait_time_s=params.wait_time_s,
                homing_force_N=params.homing_force_N
            )
            self.controller.start_shock_profile()
            ui.notify(f'Shock profile started: {profile_name} ({repetitions} rounds)', type='positive')
        else:
            ui.notify(f'Failed to load profile: {profile_name}', type='negative')

    def _load_profile_operator(self, name: str):
        """Load shock profile in operator mode"""
        if not name:
            return

        if self.controller.load_shock_profile(name):
            ui.notify(f'Profile loaded: {name}', type='positive')
        else:
            ui.notify('Failed to load profile', type='negative')

    def _refresh_profiles_operator(self):
        """Refresh the list of available profiles for operator mode"""
        if not self.operator_profile_select:
            return

        profiles = self.controller.list_shock_profiles()
        if profiles:
            self.operator_profile_select.options = profiles
            # Set default to SCAR 308 profile if available
            if '1000 G Shock - SCAR 308' in profiles:
                self.operator_profile_select.value = '1000 G Shock - SCAR 308'
            elif profiles:
                self.operator_profile_select.value = profiles[0]
        else:
            self.operator_profile_select.options = []
        self.operator_profile_select.update()
