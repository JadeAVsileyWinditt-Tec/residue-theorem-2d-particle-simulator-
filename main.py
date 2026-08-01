import time
import pygame
import numpy as np
import torch
from engine.contour import ContourPolygon

# Select NVIDIA CUDA GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Initialize Pygame ---
pygame.init()
WIDTH, HEIGHT = 900, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("NVIDIA CUDA Physics Engine - Higher-Order Poles & Discrepancy Counter")
clock = pygame.time.Clock()

# Color Palette
BG_COLOR = (15, 20, 28)
CONTOUR_COLOR = (0, 225, 160)
INSIDE_PARTICLE = (255, 85, 115)
OUTSIDE_PARTICLE = (100, 116, 139)
TBU_PARTICLE_COLOR = (255, 200, 80)
DIPOLE_COLOR = (0, 180, 255)
QUADRUPOLE_COLOR = (200, 100, 255)
TEXT_COLOR = (240, 240, 245)
GRID_COLOR = (28, 36, 48)
AXIS_COLOR = (60, 75, 95)
AIRFOIL_COLOR = (255, 180, 50)
SPHERE_WIRE_COLOR = (50, 70, 100)
HORIZON_COLOR = (255, 80, 80)
BENCHMARK_COLOR = (0, 255, 200)
ERROR_COLOR = (255, 120, 120)

CENTER_X, CENTER_Y = WIDTH // 2, HEIGHT // 2
SCALE = 80.0

# --- TBU Framework Parameters ---
EPS_TBU = 0.05
DELTA_TBU = 0.08
K_TBU = 4.0
R_HORIZON = EPS_TBU


# --- Mathematical Helper Functions ---
def S_potential(r: float) -> float:
    r = max(r, 1e-4)
    return -1.0 / r + EPS_TBU / (2.0 * r**2)


def dS_dr(r: float) -> float:
    r = max(r, 1e-4)
    return 1.0 / (r**2) - EPS_TBU / (r**3)


def B_delta(r: float) -> float:
    S_crit = S_potential(R_HORIZON)
    val = -(S_potential(r) - S_crit) / DELTA_TBU
    val = np.clip(val, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(val))


def joukowski_transform(z: complex, c: float = 1.0) -> complex:
    if abs(z) < 1e-4:
        return z
    return z + (c**2) / z


def project_to_riemann_sphere(z: complex, angle_y: float) -> tuple[int, int]:
    x, y = z.real, z.imag
    denom = x**2 + y**2 + 1.0
    R = 180.0
    X = (2.0 * x) / denom
    Y = (2.0 * y) / denom
    Z = (x**2 + y**2 - 1.0) / denom
    cos_a, sin_a = np.cos(angle_y), np.sin(angle_y)
    X_rot = X * cos_a + Z * sin_a
    return int(CENTER_X + X_rot * R), int(CENTER_Y - Y * R)


def screen_to_complex(x: int, y: int) -> complex:
    return complex((x - CENTER_X) / SCALE, -(y - CENTER_Y) / SCALE)


def complex_to_screen(z: complex) -> tuple[int, int]:
    return int(CENTER_X + z.real * SCALE), int(CENTER_Y - z.imag * SCALE)


class ParticleManagerGPU:
    """Handles Complex Vortex Dynamics, Higher-Order Poles, and TBU Keplerian Physics."""

    def __init__(self):
        self.positions = torch.empty((0,), dtype=torch.complex64, device=device)
        self.residues = torch.empty((0,), dtype=torch.complex64, device=device)
        self.pole_orders = torch.empty((0,), dtype=torch.int32, device=device)
        self.tbu_states = torch.empty((0, 4), dtype=torch.float32, device=device)

    def add_particle(self, z: complex, residue: complex, pole_order: int = 1, vr0: float = -0.1, l0: float = 0.25):
        new_z = torch.tensor([z], dtype=torch.complex64, device=device)
        new_res = torch.tensor([residue], dtype=torch.complex64, device=device)
        new_order = torch.tensor([pole_order], dtype=torch.int32, device=device)

        self.positions = torch.cat([self.positions, new_z])
        self.residues = torch.cat([self.residues, new_res])
        self.pole_orders = torch.cat([self.pole_orders, new_order])

        r0 = max(abs(z), 0.05)
        theta0 = np.angle(z)
        new_tbu = torch.tensor([[r0, vr0, theta0, l0]], dtype=torch.float32, device=device)
        self.tbu_states = torch.cat([self.tbu_states, new_tbu])

    def spawn_benchmark_cluster(self, count: int = 1000):
        angles = np.random.uniform(0, 2 * np.pi, count)
        radii = np.random.uniform(0.3, 3.5, count)
        z_pts = radii * np.exp(1j * angles)

        res_pts = np.random.choice([-1.0, 1.0, 0.5, -0.5], count) + 1j * np.random.choice([0.0, 0.25, -0.25], count)
        orders = np.random.choice([1, 2, 3], count, p=[0.7, 0.2, 0.1])
        vrs = np.random.uniform(-0.3, -0.05, count)
        ls = np.random.choice([0.1, 0.2, 0.3, 0.7], count)

        self.positions = torch.tensor(z_pts, dtype=torch.complex64, device=device)
        self.residues = torch.tensor(res_pts, dtype=torch.complex64, device=device)
        self.pole_orders = torch.tensor(orders, dtype=torch.int32, device=device)

        tbu_arr = np.column_stack([radii, vrs, angles, ls])
        self.tbu_states = torch.tensor(tbu_arr, dtype=torch.float32, device=device)

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

    def update_tbu_keplerian_physics(self, dt: float):
        N = self.tbu_states.shape[0]
        if N == 0:
            return

        for i in range(N):
            r, vr, theta, l = self.tbu_states[i].cpu().numpy()
            r_val = max(r, 0.02)
            ds_dr = dS_dr(r_val)
            b_val = B_delta(r_val)

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
            self.positions[i] = torch.tensor(z_new, dtype=torch.complex64, device=device)


# --- Setup Simulation Objects ---
particles_gpu = ParticleManagerGPU()

particles_gpu.add_particle(complex(1.8, 1.2), 1.0 + 0.0j, pole_order=1, vr0=-0.2, l0=0.25)
particles_gpu.add_particle(complex(-1.5, -1.0), 0.8 + 0.0j, pole_order=2, vr0=-0.1, l0=0.3)
particles_gpu.add_particle(complex(0.5, 2.0), -1.2 + 0.0j, pole_order=3, vr0=-0.3, l0=0.15)

contour = ContourPolygon()
font = pygame.font.SysFont("Consolas", 14)
font_bold = pygame.font.SysFont("Consolas", 15, bold=True)

# Default circular contour loop
default_center = 0.0 + 0.0j
default_radius = 2.0
for angle in np.linspace(0, 2 * np.pi, 60, endpoint=False):
    contour.add_point(default_center + default_radius * np.exp(1j * angle))
contour.close()

running = True
drawing_contour = False
joukowski_mode = False
riemann_mode = False
tbu_kepler_mode = False
active_pole_order = 1
sphere_rotation_angle = 0.0

while running:
    step_start_time = time.perf_counter()
    dt = clock.tick(60) / 1000.0
    screen.fill(BG_COLOR)
    sphere_rotation_angle += dt * 0.5

    bounds = (-CENTER_X / SCALE, CENTER_X / SCALE, -CENTER_Y / SCALE, CENTER_Y / SCALE)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:
                active_pole_order = 1
            elif event.key == pygame.K_2:
                active_pole_order = 2
            elif event.key == pygame.K_3:
                active_pole_order = 3

            elif event.key == pygame.K_k:
                tbu_kepler_mode = not tbu_kepler_mode
                if tbu_kepler_mode:
                    joukowski_mode = False
                    riemann_mode = False

            elif event.key == pygame.K_j:
                joukowski_mode = not joukowski_mode
                if joukowski_mode:
                    tbu_kepler_mode = False
                    riemann_mode = False

            elif event.key == pygame.K_s:
                riemann_mode = not riemann_mode
                if riemann_mode:
                    joukowski_mode = False
                    tbu_kepler_mode = False

            elif event.key == pygame.K_b:
                particles_gpu.spawn_benchmark_cluster(1000)

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_z = screen_to_complex(*event.pos)
            if event.button == 1:
                rand_res = complex(np.random.choice([-1.0, 1.0, 0.5]), np.random.choice([0.0, 0.5, -0.5]))
                rand_l = float(np.random.choice([0.15, 0.25, 0.35, 0.7]))
                particles_gpu.add_particle(mouse_z, rand_res, pole_order=active_pole_order, vr0=-0.2, l0=rand_l)

            elif event.button == 3:
                contour.clear()
                contour.add_point(mouse_z)
                drawing_contour = True

        elif event.type == pygame.MOUSEMOTION and drawing_contour:
            mouse_z = screen_to_complex(*event.pos)
            if len(contour.points) > 0 and abs(mouse_z - contour.points[-1]) > 0.1:
                contour.add_point(mouse_z)

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 3 and drawing_contour:
                drawing_contour = False
                contour.close()

    # --- GPU Physics Step ---
    if tbu_kepler_mode:
        particles_gpu.update_tbu_keplerian_physics(dt)
    else:
        particles_gpu.update_vortex_physics(bounds, dt)

    cpu_positions = particles_gpu.positions.cpu().numpy()
    cpu_residues = particles_gpu.residues.cpu().numpy()
    cpu_orders = particles_gpu.pole_orders.cpu().numpy()
    cpu_tbu_states = particles_gpu.tbu_states.cpu().numpy()

    # --- Mathematical Verification: Residue vs. Discrete Path Integration ---
    sum_residues = 0.0 + 0.0j
    for z, res, order in zip(cpu_positions, cpu_residues, cpu_orders):
        if order == 1 and contour.contains(complex(z)):
            sum_residues += complex(res)

    contour_integral_theoretical = 2 * np.pi * 1j * sum_residues

    # Compute discrete path line integral along closed contour C: ∮_C V(z) dz
    contour_integral_empirical = 0.0 + 0.0j
    if contour.is_closed and len(contour.points) >= 3:
        n_pts = len(contour.points)
        for i in range(n_pts):
            z_start = contour.points[i]
            z_end = contour.points[(i + 1) % n_pts]
            z_mid = 0.5 * (z_start + z_end)
            dz = z_end - z_start

            field_val = 0.0 + 0.0j
            for z_p, res_p, ord_p in zip(cpu_positions, cpu_residues, cpu_orders):
                diff = z_mid - complex(z_p)
                if abs(diff) > 0.05:
                    field_val += complex(res_p) / (diff ** int(ord_p))

            contour_integral_empirical += field_val * dz

    numerical_discrepancy = abs(contour_integral_theoretical - contour_integral_empirical)

    def map_to_screen(z: complex) -> tuple[int, int]:
        if riemann_mode:
            return project_to_riemann_sphere(z, sphere_rotation_angle)
        elif joukowski_mode:
            return complex_to_screen(joukowski_transform(z))
        else:
            return complex_to_screen(z)

    # --- Rendering Grid & Field Vectors ---
    if riemann_mode:
        pygame.draw.circle(screen, SPHERE_WIRE_COLOR, (CENTER_X, CENTER_Y), 180, 1)
        pygame.draw.ellipse(screen, SPHERE_WIRE_COLOR, (CENTER_X - 180, CENTER_Y - 50, 360, 100), 1)
    else:
        for x in range(0, WIDTH, int(SCALE)):
            pygame.draw.line(screen, GRID_COLOR, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, int(SCALE)):
            pygame.draw.line(screen, GRID_COLOR, (0, y), (WIDTH, y))

        pygame.draw.line(screen, AXIS_COLOR, (CENTER_X, 0), (CENTER_X, HEIGHT), 2)
        pygame.draw.line(screen, AXIS_COLOR, (0, CENTER_Y), (WIDTH, CENTER_Y), 2)

        if tbu_kepler_mode:
            horizon_px_radius = int(R_HORIZON * SCALE)
            pygame.draw.circle(screen, HORIZON_COLOR, (CENTER_X, CENTER_Y), horizon_px_radius, 2)

        GRID_STEP = 35
        for gx in range(0, WIDTH, GRID_STEP):
            for gy in range(0, HEIGHT, GRID_STEP):
                gz = screen_to_complex(gx, gy)
                v_z = 0.0 + 0.0j

                if not tbu_kepler_mode:
                    for z, res, order in zip(cpu_positions, cpu_residues, cpu_orders):
                        diff = gz - complex(z)
                        if abs(diff) > 0.15:
                            v_z += complex(res) / (diff ** int(order))
                else:
                    r_gz = max(abs(gz), 0.05)
                    f_rad = -dS_dr(r_gz) * (1.0 + K_TBU * (1.0 - B_delta(r_gz)))
                    v_z = complex(f_rad * np.cos(np.angle(gz)), f_rad * np.sin(np.angle(gz)))

                mag = abs(v_z)
                if mag > 1e-4:
                    direction = (v_z / mag) * min(mag * 5.0, 12.0)
                    start_px, start_py = map_to_screen(gz)
                    end_px = start_px + int(direction.real)
                    end_py = start_py - int(direction.imag)

                    alpha_col = min(int(mag * 50), 120)
                    field_color = (0, alpha_col + 50, alpha_col + 100)
                    pygame.draw.line(screen, field_color, (start_px, start_py), (end_px, end_py), 1)

    # Contour Loop C
    if len(contour.points) >= 2:
        screen_pts = [map_to_screen(z) for z in contour.points]
        c_color = AIRFOIL_COLOR if joukowski_mode else CONTOUR_COLOR
        if contour.is_closed:
            pygame.draw.polygon(screen, c_color, screen_pts, 2)
        else:
            pygame.draw.lines(screen, c_color, False, screen_pts, 2)

    # Render Particles
    for i, (z, res, order) in enumerate(zip(cpu_positions, cpu_residues, cpu_orders)):
        z_comp = complex(z)
        px, py = map_to_screen(z_comp)

        if order == 2:
            color = DIPOLE_COLOR
        elif order == 3:
            color = QUADRUPOLE_COLOR
        else:
            color = INSIDE_PARTICLE if contour.contains(z_comp) else OUTSIDE_PARTICLE

        r_size = 4 if len(cpu_positions) > 100 else 7
        pygame.draw.circle(screen, color, (px, py), r_size)

        if len(cpu_positions) <= 15:
            pole_label = "Mono" if order == 1 else ("Dipole" if order == 2 else "Quad")
            lbl = font.render(f"{pole_label}", True, color)
            screen.blit(lbl, (px + 8, py - 8))

    # --- Metrics & HUD Overlay ---
    step_time_ms = (time.perf_counter() - step_start_time) * 1000.0
    current_fps = clock.get_fps()

    if device.type == "cuda":
        vram_allocated = torch.cuda.memory_allocated() / (1024**2)
        vram_reserved = torch.cuda.memory_reserved() / (1024**2)
        gpu_str = f"GPU Memory: {vram_allocated:.1f} MB (Alloc) / {vram_reserved:.1f} MB (Res)"
    else:
        gpu_str = "Compute Device: CPU (Fallback)"

    active_order_str = "Monopole (1/z)" if active_pole_order == 1 else ("Dipole (1/z²)" if active_pole_order == 2 else "Quadrupole (1/z³)")

    hud_left = [
        f"ACTIVE SPAWN MODE: {active_order_str} [Keys 1, 2, 3]",
        f"Theoretical Residue (2πi ∑Res): {contour_integral_theoretical.real:+.2f} {contour_integral_theoretical.imag:+.2f}i",
        f"Empirical Path Integral (∮ V dz): {contour_integral_empirical.real:+.2f} {contour_integral_empirical.imag:+.2f}i",
        f"Numerical Discrepancy Error: {numerical_discrepancy:.5f}",
        "[1] Mono | [2] Dipole | [3] Quad | [K] TBU Tarpit | [J] Airfoil | [S] Sphere | [B] Bench",
    ]

    for idx, text_str in enumerate(hud_left):
        if idx == 0:
            col = (0, 200, 255)
        elif idx == 3:
            col = ERROR_COLOR
        else:
            col = TEXT_COLOR if idx < 3 else (140, 150, 170)
        txt_surface = font.render(text_str, True, col)
        screen.blit(txt_surface, (20, 20 + idx * 22))

    hud_right = [
        f"HARDWARE: {str(device).upper()}",
        f"PERFORMANCE: {current_fps:.1f} FPS ({step_time_ms:.2f} ms/frame)",
        f"PARTICLE COUNT: {len(cpu_positions)} Active Tensors",
        gpu_str,
    ]

    for idx, text_str in enumerate(hud_right):
        col = BENCHMARK_COLOR if idx == 1 else (180, 190, 210)
        txt_surface = font_bold.render(text_str, True, col)
        rect = txt_surface.get_rect(topright=(WIDTH - 20, 20 + idx * 22))
        screen.blit(txt_surface, rect)

    pygame.display.flip()

pygame.quit()
