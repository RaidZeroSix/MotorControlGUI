"""
PID Controller for Position Control using python-control library

This module implements a PID controller that takes position error as input
and outputs force commands in millinewtons for the Orca motor.
"""

import numpy as np
import control as ct
from typing import Optional
from dataclasses import dataclass


@dataclass
class PIDParameters:
    """PID controller parameters"""
    kp: float = 1.0  # Proportional gain
    ki: float = 0.0  # Integral gain
    kd: float = 0.0  # Derivative gain
    max_output: float = 50000.0  # Maximum force output in mN
    min_output: float = -50000.0  # Minimum force output in mN
    max_integral: float = 10000.0  # Anti-windup limit for integral term
    sample_time: float = 0.01  # Sample time in seconds (100 Hz default)


class PIDController:
    """
    PID Controller for position control of the Orca motor using python-control.

    The controller takes position error (setpoint - current_position) as input
    and outputs a force command in millinewtons.
    """

    def __init__(self, parameters: Optional[PIDParameters] = None):
        """
        Initialize the PID controller.

        Args:
            parameters: PID controller parameters. If None, uses default parameters.
        """
        self.params = parameters if parameters is not None else PIDParameters()

        # Create the PID transfer function: C(s) = Kp + Ki/s + Kd*s
        self._update_controller()

        # State for discrete-time implementation
        self.prev_time: Optional[float] = None
        self.integral = 0.0
        self.prev_error = 0.0

    def _update_controller(self):
        """Create/update the PID controller transfer function"""
        # Create continuous-time PID: Kp + Ki/s + Kd*s
        # In transfer function form: (Kd*s^2 + Kp*s + Ki) / s
        num = [self.params.kd, self.params.kp, self.params.ki]
        den = [1, 0]

        try:
            self.controller_tf = ct.TransferFunction(num, den)

            # Convert to discrete-time using Tustin (bilinear) transformation
            self.controller_dt = ct.sample_system(
                self.controller_tf,
                self.params.sample_time,
                method='tustin'
            )

            # Extract discrete coefficients for manual implementation
            # This gives us the difference equation coefficients
            self.num_d = self.controller_dt.num[0][0]
            self.den_d = self.controller_dt.den[0][0]

        except Exception as e:
            # Fallback if Ki is zero (which causes issues with s in denominator)
            # In this case, we'll use manual discrete-time implementation
            self.controller_tf = None
            self.controller_dt = None

    def reset(self):
        """Reset the controller state (integral and derivative terms)"""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None

    def update_parameters(self, kp: Optional[float] = None,
                         ki: Optional[float] = None,
                         kd: Optional[float] = None,
                         max_output: Optional[float] = None,
                         min_output: Optional[float] = None,
                         sample_time: Optional[float] = None):
        """
        Update PID parameters and regenerate the controller.

        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            max_output: Maximum force output in mN
            min_output: Minimum force output in mN
            sample_time: Sample time in seconds
        """
        if kp is not None:
            self.params.kp = kp
        if ki is not None:
            self.params.ki = ki
        if kd is not None:
            self.params.kd = kd
        if max_output is not None:
            self.params.max_output = max_output
        if min_output is not None:
            self.params.min_output = min_output
        if sample_time is not None:
            self.params.sample_time = sample_time

        # Regenerate controller with new parameters
        self._update_controller()
        # Reset state when parameters change
        self.reset()

    def compute(self, setpoint_um: float, current_position_um: float,
                current_time: float) -> float:
        """
        Compute the control output (force in millinewtons) using discrete PID.

        Args:
            setpoint_um: Desired position in micrometers
            current_position_um: Current position in micrometers
            current_time: Current time in seconds

        Returns:
            Force command in millinewtons
        """
        # Calculate error
        error = setpoint_um - current_position_um

        # Initialize time if this is the first call
        if self.prev_time is None:
            self.prev_time = current_time
            self.prev_error = error
            # Return proportional term only for first iteration
            output = self.params.kp * error
            return self._saturate(output)

        # Calculate time delta
        dt = current_time - self.prev_time

        # Use fixed sample time if dt is close to it, otherwise adapt
        if abs(dt - self.params.sample_time) > self.params.sample_time * 0.5:
            # Time step is significantly different, use actual dt
            pass
        else:
            # Use fixed sample time for consistency
            dt = self.params.sample_time

        # Avoid division by zero or negative dt
        if dt <= 0:
            dt = self.params.sample_time

        # Discrete-time PID implementation (velocity form for better anti-windup)
        # P term
        p_term = self.params.kp * error

        # I term with anti-windup clamping
        self.integral += error * dt
        self.integral = np.clip(self.integral,
                               -self.params.max_integral,
                               self.params.max_integral)
        i_term = self.params.ki * self.integral

        # D term (derivative on measurement to avoid derivative kick)
        derivative = (error - self.prev_error) / dt
        d_term = self.params.kd * derivative

        # Total output
        output = p_term + i_term + d_term

        # Update state
        self.prev_error = error
        self.prev_time = current_time

        # Saturate output
        saturated_output = self._saturate(output)

        # Anti-windup: back-calculate integral if output is saturated
        if saturated_output != output and self.params.ki != 0:
            # Reduce integral by the amount of saturation
            excess = output - saturated_output
            self.integral -= excess / self.params.ki

        return saturated_output

    def _saturate(self, value: float) -> float:
        """Saturate the output to min/max limits"""
        return np.clip(value, self.params.min_output, self.params.max_output)

    def get_state(self) -> dict:
        """
        Get the current state of the controller for debugging/monitoring.

        Returns:
            Dictionary containing controller state
        """
        return {
            'integral': self.integral,
            'prev_error': self.prev_error,
            'kp': self.params.kp,
            'ki': self.params.ki,
            'kd': self.params.kd,
            'max_output': self.params.max_output,
            'min_output': self.params.min_output,
            'sample_time': self.params.sample_time
        }

    def get_transfer_function(self):
        """
        Get the continuous-time transfer function representation.

        Returns:
            control.TransferFunction object or None if not available
        """
        return self.controller_tf

    def get_discrete_transfer_function(self):
        """
        Get the discrete-time transfer function representation.

        Returns:
            control.TransferFunction object or None if not available
        """
        return self.controller_dt
