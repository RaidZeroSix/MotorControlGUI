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
        # Initialize controller at 1000 Hz (1 kHz)
        # Can be set to 0 for maximum speed (no sleep between iterations)
        self.controller = MotorController(update_rate_hz=1000.0)

        # UI state
        self.connected = False

        # Data for plotting
        self.max_plot_points = 200
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

        # Update charts (ECharts format: array of [x, y] pairs)
        if self.position_chart and len(self.time_data) > 0:
            self.position_chart.options['series'][0]['data'] = [
                [t, p] for t, p in zip(self.time_data, self.position_data)
            ]
            self.position_chart.update()

        if self.force_chart and len(self.time_data) > 0:
            self.force_chart.options['series'][0]['data'] = [
                [t, f] for t, f in zip(self.time_data, self.force_data)
            ]
            self.force_chart.update()

    def create_ui(self):
        """Create the main UI"""
        ui.page_title('Pixels On Target')

        with ui.header().classes('items-center justify-between'):
            ui.label('Pixels On Target').classes('text-h4')
            self.status_label = ui.label('Status: Disconnected').classes('text-red-600')

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

                ui.button('Connect', on_click=lambda: self._connect(
                    port_select.value, int(baud_input.value)
                )).props('color=green')
                ui.button('Disconnect', on_click=self._disconnect).props('color=red')

                ui.separator().props('vertical')

                ui.button('Start Loop', on_click=self._start_control).props('color=primary')
                ui.button('Stop Loop', on_click=self._stop_control).props('color=orange')
                ui.button('E-STOP', on_click=self._emergency_stop).props('color=red')

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

        # Third row - Control setpoints
        with ui.card().classes('w-full'):
            ui.label('Control Setpoints').classes('text-h6')
            with ui.row().classes('w-full items-end gap-6'):
                # Mode selection
                mode_select = ui.select(
                    ['Sleep', 'Force Direct', 'Position'],
                    value='Sleep',
                    label='Mode'
                ).style('width: 140px')
                mode_select.on_value_change(lambda e: self._set_mode(e.value))

                ui.separator().props('vertical')

                # Force control
                force_input = ui.number('Force (N)', value=0.0, step=0.1, format='%.2f').style('width: 120px')
                ui.button('Set Force', on_click=lambda: self._set_force(force_input.value)).props('size=sm')

                ui.separator().props('vertical')

                # Position control
                position_input = ui.number('Position (mm)', value=35.0, step=0.1, format='%.2f').style('width: 120px')
                ui.button('Set Position', on_click=lambda: self._set_position(position_input.value)).props('size=sm')

                ui.separator().props('vertical')

                # PID tuning
                kp_input = ui.number('Kp', value=0.1, step=0.01, format='%.3f').style('width: 100px')
                ki_input = ui.number('Ki', value=0.01, step=0.001, format='%.4f').style('width: 100px')
                kd_input = ui.number('Kd', value=0.005, step=0.001, format='%.4f').style('width: 100px')
                ui.button('Update PID', on_click=lambda: self._update_pid(
                    kp_input.value, ki_input.value, kd_input.value
                )).props('size=sm')

        # Bottom row - Charts side by side
        with ui.row().classes('w-full gap-4'):
            with ui.card().classes('w-1/2'):
                ui.label('Position vs Time').classes('text-h6')
                self.position_chart = ui.echart({
                    'xAxis': {'type': 'value', 'name': 'Time (s)'},
                    'yAxis': {'type': 'value', 'name': 'Position (mm)'},
                    'series': [{
                        'type': 'line',
                        'name': 'Position',
                        'data': [],
                        'smooth': True
                    }],
                    'tooltip': {'trigger': 'axis'},
                    'legend': {'data': ['Position']}
                }).classes('w-full h-80')

            with ui.card().classes('w-1/2'):
                ui.label('Force vs Time').classes('text-h6')
                self.force_chart = ui.echart({
                    'xAxis': {'type': 'value', 'name': 'Time (s)'},
                    'yAxis': {'type': 'value', 'name': 'Force (N)'},
                    'series': [{
                        'type': 'line',
                        'name': 'Force',
                        'data': [],
                        'smooth': True
                    }],
                    'tooltip': {'trigger': 'axis'},
                    'legend': {'data': ['Force']}
                }).classes('w-full h-80')

        # Start periodic UI update timer
        ui.timer(0.1, self._update_ui)

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
        """Connect to motor"""
        if not port:
            ui.notify('Please select a serial port', type='warning')
            return

        if self.controller.connect(port, baud_rate):
            ui.notify(f'Connected to {port}', type='positive')
            self.connected = True
        else:
            ui.notify('Connection failed', type='negative')

    def _disconnect(self):
        """Disconnect from motor"""
        self.controller.disconnect()
        ui.notify('Disconnected', type='info')
        self.connected = False

    def _start_control(self):
        """Start control loop"""
        if self.controller.start_control_loop():
            ui.notify('Control loop started', type='positive')
        else:
            ui.notify('Failed to start control loop', type='negative')

    def _stop_control(self):
        """Stop control loop"""
        self.controller.stop_control_loop()
        ui.notify('Control loop stopped', type='warning')

    def _emergency_stop(self):
        """Emergency stop"""
        self.controller.emergency_stop()
        ui.notify('EMERGENCY STOP ACTIVATED', type='negative')

    def _set_mode(self, mode_str: str):
        """Set control mode"""
        mode_map = {
            'Sleep': ControlMode.SLEEP,
            'Force Direct': ControlMode.FORCE_DIRECT,
            'Position': ControlMode.POSITION
        }
        mode = mode_map.get(mode_str, ControlMode.SLEEP)
        self.controller.set_control_mode(mode)
        ui.notify(f'Mode set to {mode_str}', type='info')

    def _set_force(self, force_n: float):
        """Set force setpoint"""
        force_mN = force_n * 1000.0  # Convert N to mN
        self.controller.set_force_setpoint(force_mN)
        ui.notify(f'Force setpoint: {force_n:.2f} N', type='info')

    def _set_position(self, position_mm: float):
        """Set position setpoint"""
        position_um = position_mm * 1000.0  # Convert mm to um
        self.controller.set_position_setpoint(position_um)
        ui.notify(f'Position setpoint: {position_mm:.2f} mm', type='info')

    def _update_pid(self, kp: float, ki: float, kd: float):
        """Update PID parameters"""
        self.controller.update_pid_parameters(kp=kp, ki=ki, kd=kd)
        ui.notify(f'PID updated: Kp={kp:.3f}, Ki={ki:.4f}, Kd={kd:.4f}', type='info')
