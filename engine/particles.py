import numpy as np
import torch

class ParticleManagerGPU:
    """
    Handles Complex Vortex Dynamics, Higher-Order Poles, and TBU Keplerian Physics.
    Uses PyTorch tensors for GPU acceleration when available.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.positions = torch.empty((0,), dtype=torch.complex64, device=device)
        self.residues = torch.empty((0,), dtype=torch.complex64, device=device)
        self.pole_orders = torch.empty((0,), dtype=torch.int32, device=device)
        self.tbu_states = torch.empty((0, 4), dtype=torch.float32, device=device)

    def add_particle(
        self,
        z: complex,
        residue: complex,
        pole_order: int = 1,
        vr0: float = -0.1,
        l0: float = 0.25,
    ):
        new_z = torch.tensor([z], dtype=torch.complex64, device=self.device)
        new_res = torch.tensor([residue], dtype=torch.complex64, device=self.device)
        new_order = torch.tensor([pole_order], dtype=torch.int32, device=self.device)

        self.positions = torch.cat([self.positions, new_z])
        self.residues = torch.cat([self.residues, new_res])
        self.pole_orders = torch.cat([self.pole_orders, new_order])

        r0 = max(abs(z), 0.05)
        theta0 = np.angle(z)
        new_tbu = torch.tensor([[r0, vr0, theta0, l0]], dtype=torch.float32, device=self.device)
        self.tbu_states = torch.cat([self.tbu_states, new_tbu])

    def spawn_benchmark_cluster(self, count: int = 1000):
        angles = np.random.uniform(0, 2 * np.pi, count)
        radii = np.random.uniform(0.3, 3.5, count)
        z_pts = radii * np.exp(1j * angles)

        res_pts = (
            np.random.choice([-1.0, 1.0, 0.5, -0.5], count)
            + 1j * np.random.choice([0.0, 0.25, -0.25], count)
        )
        orders = np.random.choice([1, 2, 3], count, p=[0.7, 0.2, 0.1])
        vrs = np.random.uniform(-0.3, -0.05, count)
        ls = np.random.choice([0.1, 0.2, 0.3, 0.7], count)

        self.positions = torch.tensor(z_pts, dtype=torch.complex64, device=self.device)
        self.residues = torch.tensor(res_pts, dtype=torch.complex64, device=self.device)
        self.pole_orders = torch.tensor(orders, dtype=torch.int32, device=self.device)

        tbu_arr = np.column_stack([radii, vrs, angles, ls])
        self.tbu_states = torch.tensor(tbu_arr, dtype=torch.float32, device=self.device)

    def update_vortex_physics(self, bounds: tuple[float, float, float, float], dt: float):
        N = self.positions.shape[0]
        if N <= 1:
            return

        z_col = self.positions.unsqueeze(1)
        z_row = self.positions.unsqueeze(0)
        diff = z_col - z_row

        dist = torch.abs(diff)
        mask = (dist > 0.1).float()

        orders_row = self.pole_orders.unsqueeze(0)
        denom = torch.pow(torch.conj(diff + 1e-6), orders_row)

        v_matrix = (1j / (2 * np.pi)) * (self.residues.unsqueeze(0) / denom)
        v_matrix = v_matrix * mask

        v_induced = torch.sum(v_matrix, dim=1)
        self.positions += v_induced * dt

        min_re, max_re, min_im, max_im = bounds
        padding = 0.2
        re = torch.clamp(self.positions.real, min_re + padding, max_re - padding)
        im = torch.clamp(self.positions.imag, min_im + padding, max_im - padding)
        self.positions = torch.complex(re, im)

    def update_tbu_keplerian_physics(
        self,
        dt: float,
        dS_dr_func,
        B_delta_func,
        K_TBU: float,
    ):
        """
        TBU update. Requires the helper functions and K_TBU to be passed in
        so this module stays independent of main.py globals.
        """
        N = self.tbu_states.shape[0]
        if N == 0:
            return

        for i in range(N):
            r, vr, theta, l = self.tbu_states[i].cpu().numpy()
            r_val = max(r, 0.02)
            ds_dr = dS_dr_func(r_val)
            b_val = B_delta_func(r_val)

            radial_force = -ds_dr * (1.0 + K_TBU * (1.0 - b_val))
            centrifugal_force = (l**2) / (r_val**3)

            vr += (radial_force + centrifugal_force) * dt
            r += vr * dt
            theta += (l / (r_val**2)) * dt

            r = max(r, 0.02)
            z_new = complex(r * np.cos(theta), r * np.sin(theta))

            self.tbu_states[i, 0] = r
            self.tbu_states[i, 1] = vr
            self.tbu_states[i, 2] = theta
            self.positions[i] = torch.tensor(z_new, dtype=torch.complex64, device=self.device)
