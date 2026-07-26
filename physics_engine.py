import numpy as np
from flow_models import PotentialFlow

class RK4ParticleEngine:
    """Production RK4 integrator for particles moving through analytic complex fields."""
    
    def __init__(self, flow_field: PotentialFlow, mass: float = 1.0, drag: float = 0.05):
        self.field = flow_field
        self.mass = mass
        self.drag = drag

    def step_particle(self, z: complex, v: complex, dt: float) -> tuple[complex, complex]:
        """
        4th-order Runge-Kutta integration step for state vector S = [z, v]^T.
        F(z) = Force_field(z) - drag * v
        """
        def f_state(z_curr, v_curr):
            # Compute acceleration from potential flow field
            force = self.field.complex_velocity(z_curr).conjugate()
            accel = (force - self.drag * v_curr) / self.mass
            return v_curr, accel

        # RK4 Constants
        kv1, ka1 = f_state(z, v)
        kv2, ka2 = f_state(z + 0.5 * dt * kv1, v + 0.5 * dt * ka1)
        kv3, ka3 = f_state(z + 0.5 * dt * kv2, v + 0.5 * dt * ka2)
        kv4, ka4 = f_state(z + dt * kv3, v + dt * ka3)

        z_next = z + (dt / 6.0) * (kv1 + 2*kv2 + 2*kv3 + kv4)
        v_next = v + (dt / 6.0) * (ka1 + 2*ka2 + 2*ka3 + ka4)

        return z_next, v_next
