import pygame
import numpy as np
import torch

# Select NVIDIA CUDA GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Initialize Pygame ---
pygame.init()
WIDTH, HEIGHT = 900, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("NVIDIA CUDA 2D Physics Engine - TBU Keplerian Tarpit & Complex Analysis")
clock = pygame.time.Clock()

# Color Palette
BG_COLOR = (15, 20, 28)
CONTOUR_COLOR = (0, 225, 160)
INSIDE_PARTICLE = (255, 85, 115)
OUTSIDE_PARTICLE = (100, 116, 139)
TBU_PARTICLE_COLOR = (255, 200, 80)
TEXT_COLOR = (240, 240, 245)
GRID_COLOR = (28, 36, 48)
AXIS_COLOR = (60, 75, 95)
AIRFOIL_COLOR = (255, 180, 50)
SPHERE_WIRE_COLOR = (50, 70, 100)
HORIZON_COLOR = (255, 80, 80)

CENTER_X, CENTER_Y = WIDTH // 2, HEIGHT // 2
SCALE = 80.0

# --- TBU Framework Parameters ---
EPS_TBU = 0.05       # Soft-core parameter
DELTA_TBU = 0.08     # Boundary layer thickness
K_TBU = 4.0          # Dissipative containment strength
R_HORIZON = EPS_TBU  # Critical horizon radius


# --- Mathematical Helper Functions ---
def S_potential(r: float) -> float:
    """Action potential S(r) = -1/r + eps/(2*r^2)."""
    r = max(r, 1e-4)
    return -1.0 / r + EPS_TBU / (2.0 * r**2)


def dS_dr(r: float) -> float:
    """Derivative dS/dr."""
    r = max(r, 1e-4)
    return 1.0 / (r**2) - EPS_TBU / (r**3)


def B_delta(r: float) -> float:
    """Sigmoid state measure B_delta(r)."""
    S_crit = S_potential(R_HORIZON)
    val = -(S_potential(r) - S_crit) / DELTA_TBU
    val = np.clip(val, -50.0, 50.0)  # Prevent overflow
    return 1.0 / (1.0 + np.exp(val))


def joukowski_transform(z: complex, c: float = 1.0) -> complex:
    """Conformal mapping: w = z + c^2 / z."""
    if abs(z) < 1e-4:
        return z
    return z + (c**2) / z


def project_to_riemann_sphere(z: complex, angle_y: float) -> tuple[int, int]:
    """Projects 2D complex point z onto a rotating 3D Riemann sphere surface."""
    x, y = z.real, z.imag
    denom = x**2 + y**2 + 1.0

    R = 180.0
    X = (2.0 * x) / denom
    Y = (2.0 * y) / denom
    Z = (x**2 + y**2 - 1.0) / denom

    cos_a, sin_a = np.cos(angle_y), np.sin(angle_y)
    X_rot = X * cos_a + Z * sin_a

    screen_x = int(CENTER_X + X_rot * R)
    screen_y = int(CENTER_Y - Y * R)
    return screen_x, screen_y


def screen_to_complex(x: int, y: int) -> complex:
    return complex((x - CENTER_X) / SCALE, -(y - CENTER_Y) / SCALE)


def complex_to_screen(z: complex) -> tuple[int, int]:
    return int(CENTER_X + z.real * SCALE), int(CENTER_Y - z.imag * SCALE)


class ParticleManagerGPU:
    """Handles both Complex Vortex Dynamics and TBU Keplerian Tarpit Physics."""

    def __init__(self):
        # General state
        self.positions = torch.empty((0,), dtype=torch.complex64, device=device)
        self.residues = torch.empty((0,), dtype=torch.complex64, device=device)

        # TBU Polar physics state: [r, v_r, theta, l]
        self.tbu_states = torch.empty((0, 4), dtype=torch.float32, device=device)

    def add_particle(self, z: complex, residue: complex, vr0: float = -0.1, l0: float = 0.25):
        new_z = torch.tensor([z], dtype=torch.complex64, device=device)
        new_res = torch.tensor([residue], dtype=torch.complex64, device=device)
        self.positions = torch.cat([self.positions, new_z])
        self.residues = torch.cat([self.residues, new_res])

        # Convert z to polar coordinates for TBU dynamics
        r0 = max(abs(z), 0.05)
        theta0 = np.angle(z)
        new_tbu = torch.tensor([[r0, vr0, theta0, l0]], dtype=torch.float32, device=device)
        self.tbu_states = torch.cat([self.tbu_states, new_tbu])

    def update_vortex_physics(self, bounds: tuple[float, float, float, float], dt: float):
        """Standard Complex Potential Vortex Dynamics."""
        N = self.positions.shape[0]
        if N <= 1:
            return

        z_col = self.positions.unsqueeze(1)
        z_row = self.positions.unsqueeze(0)
        diff = z_col - z_row

        dist = torch.abs(diff)
        mask = (dist > 0.1).float()

        v_matrix = (1j / (2 * np.pi)) * (self.residues.unsqueeze(0) / torch.conj(diff + 1e-6))
        v_matrix = v_matrix * mask

        v_induced = torch.sum(v_matrix, dim=1)
        self.positions += v_induced * dt

        min_re, max_re, min_im, max_im = bounds
        padding = 0.2
        re = torch.clamp(self.positions.real, min_re + padding, max_re - padding)
        im = torch.clamp(self.positions.imag, min_im + padding, max_im - padding)
        self.positions = torch.complex(re, im)

    def update_tbu_keplerian_physics(self, dt: float):
        """TBU Section 4 Keplerian Tarpit Non-Hermitian ODE Dynamics."""
        N = self.tbu_states.shape[0]
        if N == 0:
            return

        for i in range(N):
            r, vr, theta, l = self.tbu_states[i].cpu().numpy()

            r_val = max(r, 0.02)
            ds_dr = dS_dr(r_val)
            b_val = B_delta(r_val)

            # Enhanced dissipative containment force
            radial_force = -ds_dr * (1.0 + K_TBU * (1.0 - b_val))
            centrifugal_force = (l**2) / (r_val**3)

            # Euler-Cromer integration
            vr += (radial_force + centrifugal_force) * dt
            r += vr * dt
            theta += (l / (r_val**2)) * dt

            r = max(r, 0.02)  # Core collision bound
            z_new = complex(r * np.cos(theta), r * np.sin(theta))

            self.tbu_states[i, 0] = r
            self.tbu_states[i, 1] = vr
            self.tbu_states[i, 2] = theta
            self.positions[i] = torch.tensor(z_new, dtype=torch.complex64, device=device)


class ContourPolygon:
    """Custom freehand closed contour C in the complex plane."""

    def __init__(self):
        self.points: list[complex] = []
        self.is_closed = False

    def clear(self):
        self.points.clear()
        self.is_closed = False

    def add_point(self, z: complex):
        if not self.is_closed:
            self.points.append(z)

    def close(self):
        if len(self.points) >= 3:
            self.is_closed = True

    def contains(self, z: complex) -> bool:
        if not self.is_closed or len(self.points) < 3:
            return False
        n = len(self.points)
        inside = False
        x, y = z.real, z.imag
        p1 = self.points[0]
        for i in range(n + 1):
            p2 = self.points[i % n]
            if y > min(p1.imag, p2.imag):
                if y <= max(p1.imag, p2.imag):
                    if x <= max(p1.real, p2.real):
                        if p1.imag != p2.imag:
                            xinters = (y - p1.imag) * (p2.real - p1.real) / (p2.imag - p1.imag) + p1.real
                        if p1.real == p2.real or x <= xinters:
                            inside = not inside
            p1 = p2
        return inside


# --- Setup Simulation Objects ---
particles_gpu = ParticleManagerGPU()

# Default initial state
particles_gpu.add_particle(complex(1.8, 1.2), 1.0 + 0.0j, vr0=-0.2, l0=0.25)
particles_gpu.add_particle(complex(-1.5, -1.0), 0.5 + 0.5j, vr0=-0.1, l0=0.3)
particles_gpu.add_particle(complex(0.5, 2.0), -1.0 + 0.0j, vr0=-0.3, l0=0.15)

contour = ContourPolygon()
font = pygame.font.SysFont("Consolas", 15)

# Initialize default circular contour loop
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
sphere_rotation_angle = 0.0

while running:
    dt = clock.tick(60) / 1000.0
    screen.fill(BG_COLOR)
    sphere_rotation_angle += dt * 0.5

    bounds = (-CENTER_X / SCALE, CENTER_X / SCALE, -CENTER_Y / SCALE, CENTER_Y / SCALE)

    # --- Event Handling ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_k:  # Toggle Keplerian Tarpit Mode
                tbu_kepler_mode = not tbu_kepler_mode
                if tbu_kepler_mode:
                    joukowski_mode = False
                    riemann_mode = False

            elif event.key == pygame.K_j:  # Toggle Joukowski Airfoil Mode
                joukowski_mode = not joukowski_mode
                if joukowski_mode:
                    tbu_kepler_mode = False
                    riemann_mode = False

            elif event.key == pygame.K_s:  # Toggle 3D Riemann Sphere Mode
                riemann_mode = not riemann_mode
                if riemann_mode:
                    joukowski_mode = False
                    tbu_kepler_mode = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_z = screen_to_complex(*event.pos)
            if event.button == 1:
                rand_res = complex(np.random.choice([-1.0, 1.0, 0.5]), np.random.choice([0.0, 0.5, -0.5]))
                rand_l = float(np.random.choice([0.15, 0.25, 0.35, 0.7]))  # l <= 0.3 captures, l >= 0.7 escapes
                particles_gpu.add_particle(mouse_z, rand_res, vr0=-0.2, l0=rand_l)

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
    cpu_tbu_states = particles_gpu.tbu_states.cpu().numpy()

    sum_residues = 0.0 + 0.0j
    inside_count = 0

    for z, res in zip(cpu_positions, cpu_residues):
        if contour.contains(complex(z)):
            sum_residues += complex(res)
            inside_count += 1

    contour_integral = 2 * np.pi * 1j * sum_residues

    # Coordinate mapping helper
    def map_to_screen(z: complex) -> tuple[int, int]:
        if riemann_mode:
            return project_to_riemann_sphere(z, sphere_rotation_angle)
        elif joukowski_mode:
            return complex_to_screen(joukowski_transform(z))
        else:
            return complex_to_screen(z)

    # --- Rendering ---
    if riemann_mode:
        # 3D Riemann Wireframe
        pygame.draw.circle(screen, SPHERE_WIRE_COLOR, (CENTER_X, CENTER_Y), 180, 1)
        pygame.draw.ellipse(screen, SPHERE_WIRE_COLOR, (CENTER_X - 180, CENTER_Y - 50, 360, 100), 1)
    else:
        # Axes & Grid
        for x in range(0, WIDTH, int(SCALE)):
            pygame.draw.line(screen, GRID_COLOR, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, int(SCALE)):
            pygame.draw.line(screen, GRID_COLOR, (0, y), (WIDTH, y))

        pygame.draw.line(screen, AXIS_COLOR, (CENTER_X, 0), (CENTER_X, HEIGHT), 2)
        pygame.draw.line(screen, AXIS_COLOR, (0, CENTER_Y), (WIDTH, CENTER_Y), 2)

        # If in TBU mode, render the critical horizon circle r_h
        if tbu_kepler_mode:
            horizon_px_radius = int(R_HORIZON * SCALE)
            pygame.draw.circle(screen, HORIZON_COLOR, (CENTER_X, CENTER_Y), horizon_px_radius, 2)

        # Vector Field Line Grid
        GRID_STEP = 30
        for gx in range(0, WIDTH, GRID_STEP):
            for gy in range(0, HEIGHT, GRID_STEP):
                gz = screen_to_complex(gx, gy)
                v_z = 0.0 + 0.0j

                if not tbu_kepler_mode:
                    for z, res in zip(cpu_positions, cpu_residues):
                        diff = gz - complex(z)
                        if abs(diff) > 0.15:
                            v_z += complex(res) / diff
                else:
                    r_gz = max(abs(gz), 0.05)
                    f_rad = -dS_dr(r_gz) * (1.0 + K_TBU * (1.0 - B_delta(r_gz)))
                    v_z = complex(f_rad * np.cos(np.angle(gz)), f_rad * np.sin(np.angle(gz)))

                mag = abs(v_z)
                if mag > 1e-4:
                    direction = (v_z / mag) * min(mag * 5.0, 15.0)
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
    for i, (z, res) in enumerate(zip(cpu_positions, cpu_residues)):
        z_comp = complex(z)
        px, py = map_to_screen(z_comp)

        if tbu_kepler_mode:
            color = TBU_PARTICLE_COLOR
            l_val = cpu_tbu_states[i, 3] if i < len(cpu_tbu_states) else 0.25
            info_txt = f"l={l_val:.2f}"
        else:
            is_inside = contour.contains(z_comp)
            color = INSIDE_PARTICLE if is_inside else OUTSIDE_PARTICLE
            res_comp = complex(res)
            info_txt = f"{res_comp.real:+.1f}{res_comp.imag:+.1f}i"

        pygame.draw.circle(screen, color, (px, py), 8)
        lbl = font.render(info_txt, True, (170, 180, 200))
        screen.blit(lbl, (px + 10, py - 10))

    # HUD Overlay
    mode_str = (
        "TBU KEPLERIAN TARPIT CAPTURE"
        if tbu_kepler_mode
        else ("3D RIEMANN SPHERE" if riemann_mode else ("JOUKOWSKI AIRFOIL" if joukowski_mode else "STANDARD COMPLEX PLANE"))
    )

    hud_data = [
        f"Device: {str(device).upper()}  |  Mode: {mode_str}",
        f"Sum of Residues (∑ Res): {sum_residues.real:+.2f} {sum_residues.imag:+.2f}i",
        f"Contour Integral (2πi * ∑ Res): {contour_integral.real:+.2f} {contour_integral.imag:+.2f}i",
        "[K] Toggle Keplerian Tarpit  |  [J] Toggle Airfoil  |  [S] Toggle 3D Riemann",
    ]

    for idx, text_str in enumerate(hud_data):
        col = (255, 200, 80) if idx == 0 and tbu_kepler_mode else ((100, 210, 255) if idx == 2 else (TEXT_COLOR if idx < 3 else (140, 150, 170)))
        txt_surface = font.render(text_str, True, col)
        screen.blit(txt_surface, (20, 20 + idx * 24))

    pygame.display.flip()

pygame.quit()
