import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class TBUNonHermitianSimulator:
    """
    Simulates TBU dynamics under complex potential fields and a non-Hermitian Hamiltonian:
    H_eff(x) = H_0(x) - i * Gamma_delta(x)
    """
    def __init__(self, eps=0.05, delta=0.08, K=4.0, l=0.25):
        self.eps = eps
        self.delta = delta
        self.K = K
        self.l = l
        
        # Calculate critical radius (r_h) and critical action S_crit
        self.r_h = self.eps
        self.S_crit = self.S(self.r_h)

    def S(self, r):
        """Action potential S(r) with soft core."""
        return -1.0 / r + self.eps / (2.0 * r**2)

    def dS_dr(self, r):
        """Derivative of action potential with respect to r."""
        return 1.0 / (r**2) - self.eps / (r**3)

    def B_delta(self, r):
        """Logistic / sigmoid softening function B_delta(r)."""
        arg = -(self.S(r) - self.S_crit) / self.delta
        # Clip argument to prevent overflow in exp
        arg = np.clip(arg, -50, 50)
        return 1.0 / (1.0 + np.exp(arg))

    def gamma_delta(self, r):
        """Non-Hermitian dissipation term Gamma_delta(x) >= 0."""
        return self.K * (1.0 - self.B_delta(r))

    def ode_system(self, t, state):
        """
        State vector: [r, v_r, theta, phi]
        where phi tracks the integrated non-Hermitian imaginary phase/decay:
        d(phi)/dt = Gamma_delta(r)
        """
        r, v_r, theta, phi = state
        
        # Prevent non-physical or zero radius evaluation
        r = max(r, 1e-6)
        
        # Dissipative containment factor inside boundary layer
        b_val = self.B_delta(r)
        force = -self.dS_dr(r) * (1.0 + self.K * (1.0 - b_val))
        centrifugal = (self.l**2) / (r**3)
        
        dr_dt = v_r
        dvr_dt = force + centrifugal
        dtheta_dt = self.l / (r**2)
        dphi_dt = self.gamma_delta(r)  # Imaginary component integration
        
        return [dr_dt, dvr_dt, dtheta_dt, dphi_dt]

    def run_simulation(self, r0=0.45, vr0=-0.1, theta0=0.0, t_span=(0.0, 80.0), num_points=2000):
        y0 = [r0, vr0, theta0, 0.0]
        
        sol = solve_ivp(
            self.ode_system, 
            t_span, 
            y0, 
            method='RK45', 
            rtol=1e-7, 
            atol=1e-9,
            dense_output=True
        )
        
        t = np.linspace(t_span[0], sol.t[-1], num_points)
        states = sol.sol(t)
        
        r, vr, theta, phi = states[0], states[1], states[2], states[3]
        
        # Calculate effective mechanical energy
        E_eff = 0.5 * (vr**2) + 0.5 * (self.l**2) / (r**2) + self.S(r)
        # Quantum norm attenuation factor exp(-2 * phi)
        amplitude_norm = np.exp(-phi)
        
        return {
            't': t,
            'r': r,
            'vr': vr,
            'theta': theta,
            'phi': phi,
            'x': r * np.cos(theta),
            'y': r * np.sin(theta),
            'E_eff': E_eff,
            'amplitude_norm': amplitude_norm
        }


if __name__ == "__main__":
    # Initialize simulator with low angular momentum (capturing regime)
    sim = TBUNonHermitianSimulator(eps=0.05, delta=0.08, K=4.0, l=0.25)
    data = sim.run_simulation()

    # Visualization
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))

    # Plot Trajectory in 2D Space
    axs[0].plot(data['x'], data['y'], color='tab:blue', label='Trajectory')
    axs[0].set_aspect('equal')
    axs[0].set_title(f"TBU Non-Hermitian Orbit (l={sim.l}, K={sim.K})")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].grid(True)
    axs[0].legend()

    # Plot Energy and State Decay
    axs[1].plot(data['t'], data['E_eff'], label='Effective Energy E(t)', color='tab:red')
    axs[1].plot(data['t'], data['amplitude_norm'], label=r'Quantum Norm $\exp(-\phi)$', color='tab:purple', linestyle='--')
    axs[1].set_title("Energy Dissipation & Non-Hermitian Decay")
    axs[1].set_xlabel("Time t")
    axs[1].set_ylabel("Magnitude")
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.show()
