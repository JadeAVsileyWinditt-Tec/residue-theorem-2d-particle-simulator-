import numpy as np
import matplotlib.pyplot as plt
from tbu_non_hermitian_integrator import TBUNonHermitianSimulator

class EnsembleSimulator:
    """Batch solver for mapping particle capture basins across initial phase space."""
    def __init__(self, eps=0.05, delta=0.08, K=4.0, l=0.25):
        self.sim = TBUNonHermitianSimulator(eps=eps, delta=delta, K=K, l=l)

    def run_ensemble(self, r0_range=(0.2, 0.8), vr0_range=(-0.5, 0.1), grid_size=15):
        r0_vals = np.linspace(r0_range[0], r0_range[1], grid_size)
        vr0_vals = np.linspace(vr0_range[0], vr0_range[1], grid_size)
        
        capture_grid = np.zeros((grid_size, grid_size))

        print(f"Running batch ensemble simulation ({grid_size}x{grid_size} grid)...")
        for i, r0 in enumerate(r0_vals):
            for j, vr0 in enumerate(vr0_vals):
                data = self.sim.run_simulation(r0=r0, vr0=vr0, t_span=(0.0, 50.0), num_points=500)
                # Captured condition: final effective energy < 0
                if data['E_eff'][-1] < 0:
                    capture_grid[j, i] = 1  # 1 = Captured / Trapped

        return r0_vals, vr0_vals, capture_grid

if __name__ == "__main__":
    ensemble = EnsembleSimulator()
    r0s, vrs, grid = ensemble.run_ensemble(grid_size=12)

    plt.figure(figsize=(7, 6))
    plt.imshow(grid, extent=[r0s[0], r0s[-1], vrs[0], vrs[-1]], origin='lower', cmap='Blues', aspect='auto')
    plt.colorbar(label='Captured (1) vs Escaped (0)')
    plt.xlabel('Initial Radius r_0')
    plt.ylabel('Initial Radial Velocity v_r0')
    plt.title(f'TBU Tarpit Capture Basin (l={ensemble.sim.l}, K={ensemble.sim.K})')
    plt.tight_layout()
    plt.show()
