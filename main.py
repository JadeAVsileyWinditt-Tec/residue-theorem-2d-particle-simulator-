import pygame
import numpy as np
import torch

# Select NVIDIA CUDA GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running simulation on device: {device}")

# --- Initialize Pygame ---
pygame.init()
WIDTH, HEIGHT = 900, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("NVIDIA CUDA-Accelerated 2D Complex Particle Simulator")
clock = pygame.time.Clock()

# Color Palette
BG_COLOR = (15, 20, 28)
CONTOUR_COLOR = (0, 225, 160)
INSIDE_PARTICLE = (255, 85, 115)
OUTSIDE_PARTICLE = (100, 116, 139)
TEXT_COLOR = (240, 240, 245)
GRID_COLOR = (28, 36, 48)
AXIS_COLOR = (60, 75, 95)

CENTER_X, CENTER_Y = WIDTH // 2, HEIGHT // 2
SCALE = 80.0


def screen_to_complex(x: int, y: int) -> complex:
    return complex((x - CENTER_X) / SCALE, -(y - CENTER_Y) / SCALE)


def complex_to_screen(z: complex) -> tuple[int, int]:
    return int(CENTER_X + z.real * SCALE), int(CENTER_Y - z.imag * SCALE)


class ParticleManagerGPU:
    """Handles point-vortex GPU physics tensors for high-performance N-body calculations."""

    def __init__(self):
        self.positions = torch.empty((0,), dtype=torch.complex64, device=device)
        self.residues = torch.empty((0,), dtype=torch.complex64, device=device)

    def add_particle(self, z: complex, residue: complex):
        new_z = torch.tensor([z], dtype=torch.complex64, device=device)
        new_res = torch.tensor([residue], dtype=torch.complex64, device=device)
        self.positions = torch.cat([self.positions, new_z])
        self.residues = torch.cat([self.residues, new_res])

    def update_physics(self, bounds: tuple[float, float, float, float], dt: float):
        N = self.positions.shape[0]
        if N <= 1:
            return

        # Vectorized N-body distance matrix on GPU: diff[i, j] = z_i - z_j
        z_col = self.positions.unsqueeze(1)  # Shape (N, 1)
        z_row = self.positions.unsqueeze(0)  # Shape (1, N)
        diff = z_col - z_row                 # Shape (N, N)

        # Mask self-interaction to avoid division by zero
        dist = torch.abs(diff)
        mask = (dist > 0.1).float()
        
        # Calculate velocity matrix: v_ij = (i / 2pi) * (Res_j / diff_ij*)
        v_matrix = (1j / (2 * np.pi)) * (self.residues.unsqueeze(0) / torch.conj(diff + 1e-6))
        v_matrix = v_matrix * mask

        # Sum velocities across all interaction pairs
        v_induced = torch.sum(v_matrix, dim=1)

        # Step forward in time: z += v * dt
        self.positions += v_induced * dt

        # Boundary box confinement on GPU
        min_re, max_re, min_im, max_im = bounds
        padding = 0.2
        re = torch.clamp(self.positions.real, min_re + padding, max_re - padding)
        im = torch.clamp(self.positions.imag, min_im + padding, max_im - padding)
        self.positions = torch.complex(re, im)


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

# Default particles
particles_gpu.add_particle(complex(-1.5, 1.2), 1.0 + 0.0j)
particles_gpu.add_particle(complex(1.2, -1.0), 0.5 + 0.5j)
particles_gpu.add_particle(complex(0.2, 0.5), -1.0 + 0.0j)

contour = ContourPolygon()
font = pygame.font.SysFont("Consolas", 15)

# Initialize default circular contour loop
default_center = 0 + 0j
default_radius = 2.2
for angle in np.linspace(0, 2 * np.pi, 30, endpoint=False):
    contour.add_point(default_center + default_radius * np.exp(1j * angle))
contour.close()

running = True
drawing_contour = False

while running:
    dt = clock.tick(60) / 1000.0
    screen.fill(BG_COLOR)

    bounds = (-CENTER_X / SCALE, CENTER_X / SCALE, -CENTER_Y / SCALE, CENTER_Y / SCALE)

    # --- Event Handling ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_z = screen_to_complex(*event.pos)
            if event.button == 1:  # Left Click: Spawn Particle
                rand_res = complex(
                    np.random.choice([-1.0, 1.0, 0.5]),
                    np.random.choice([0.0, 0.5, -0.5]),
                )
                particles_gpu.add_particle(mouse_z, rand_res)

            elif event.button == 3:  # Right Click: Draw Contour
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

    # --- GPU Physics Update ---
    particles_gpu.update_physics(bounds, dt)

    # Transfer GPU results to CPU for rendering & residue summation
    cpu_positions = particles_gpu.positions.cpu().numpy()
    cpu_residues = particles_gpu.residues.cpu().numpy()

    sum_residues = 0.0 + 0.0j
    inside_count = 0

    for z, res in zip(cpu_positions, cpu_residues):
        if contour.contains(complex(z)):
            sum_residues += complex(res)
            inside_count += 1

    contour_integral = 2 * np.pi * 1j * sum_residues

    # --- Rendering ---
    # 1. Grid & Axes
    for x in range(0, WIDTH, int(SCALE)):
        pygame.draw.line(screen, GRID_COLOR, (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, int(SCALE)):
        pygame.draw.line(screen, GRID_COLOR, (0, y), (WIDTH, y))

    pygame.draw.line(screen, AXIS_COLOR, (CENTER_X, 0), (CENTER_X, HEIGHT), 2)
    pygame.draw.line(screen, AXIS_COLOR, (0, CENTER_Y), (WIDTH, CENTER_Y), 2)

    # 2. Vector Field Line Grid
    GRID_STEP = 30
    for gx in range(0, WIDTH, GRID_STEP):
        for gy in range(0, HEIGHT, GRID_STEP):
            gz = screen_to_complex(gx, gy)
            v_z = 0.0 + 0.0j
            for z, res in zip(cpu_positions, cpu_residues):
                diff = gz - complex(z)
                if abs(diff) > 0.15:
                    v_z += complex(res) / diff

            mag = abs(v_z)
            if mag > 1e-4:
                direction = (v_z / mag) * min(mag * 5.0, 15.0)
                end_x = gx + int(direction.real)
                end_y = gy - int(direction.imag)
                alpha_col = min(int(mag * 50), 120)
                field_color = (0, alpha_col + 50, alpha_col + 100)
                pygame.draw.line(screen, field_color, (gx, gy), (end_x, end_y), 1)

    # 3. Contour Polygon C
    if len(contour.points) >= 2:
        screen_pts = [complex_to_screen(z) for z in contour.points]
        if contour.is_closed:
            pygame.draw.polygon(screen, CONTOUR_COLOR, screen_pts, 2)
        else:
            pygame.draw.lines(screen, CONTOUR_COLOR, False, screen_pts, 2)

    # 4. Particles
    for z, res in zip(cpu_positions, cpu_residues):
        z_comp = complex(z)
        res_comp = complex(res)
        px, py = complex_to_screen(z_comp)
        is_inside = contour.contains(z_comp)
        color = INSIDE_PARTICLE if is_inside else OUTSIDE_PARTICLE

        pygame.draw.circle(screen, color, (px, py), 8)
        res_text = f"{res_comp.real:+.1f}{res_comp.imag:+.1f}i"
        lbl = font.render(res_text, True, (170, 180, 200))
        screen.blit(lbl, (px + 10, py - 10))

    # 5. HUD Dashboard
    hud_data = [
        f"Device: {str(device).upper()}  |  Poles: {len(cpu_positions)}  |  Inside: {inside_count}",
        f"Sum of Residues (∑ Res): {sum_residues.real:+.2f} {sum_residues.imag:+.2f}i",
        f"Contour Integral (2πi * ∑ Res): {contour_integral.real:+.2f} {contour_integral.imag:+.2f}i",
        "[Left-Click] Add Pole  |  [Right-Click + Drag] Draw Closed Contour",
    ]

    for idx, text_str in enumerate(hud_data):
        col = (100, 210, 255) if idx == 2 else (TEXT_COLOR if idx < 3 else (140, 150, 170))
        txt_surface = font.render(text_str, True, col)
        screen.blit(txt_surface, (20, 20 + idx * 24))

    pygame.display.flip()

pygame.quit()
