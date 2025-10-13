#!/usr/bin/env python3
"""
Main entry point for the Orca Motor Control GUI

Launch the NiceGUI-based motor control application.
"""

from nicegui import ui
from app import MotorGUI
import sys

# Create the GUI instance globally
gui = MotorGUI()


@ui.page('/')
def index():
    """Main page"""
    gui.create_ui()


def main():
    """Main entry point"""
    print("Starting Orca Motor Control GUI...")

    try:
        # Run in browser mode (accessible at http://localhost:8080)
        ui.run(
            title='Orca Motor Control',
            port=8080,
            reload=False,
            show=True  # Automatically open browser
        )
    except KeyboardInterrupt:
        print("\nShutting down...")
        # Clean up: stop control loop and disconnect
        if gui.controller.state.running:
            gui.controller.stop_control_loop()
        if gui.controller.state.connected:
            gui.controller.disconnect()
        sys.exit(0)


if __name__ == '__main__':
    main()
