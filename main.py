import pygame
import numpy as np

# --- Initialize Pygame ---
pygame.init()
WIDTH, HEIGHT = 900, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2D Complex Particle Simulator - Residue Theorem")
clock = pygame.time.Clock()

# Color Palette
BG_COLOR = (15, 20, 28)
CONTOUR_COLOR = (0, 225, 160)
INSIDE_PARTICLE = (255, 85, 115)
OUTSIDE_PARTICLE = (100, 116, 139)
TEXT_COLOR = (240, 240, 245)
GRID_COLOR = (28, 36, 48)
AXIS_COLOR = (60, 75, 95)

# Coordinate conversion setup
CENTER_X, CENTER_Y = WIDTH // 2, HEIGHT // 2
SCALE = 80.0  # 80 pixels = 1.0 unit in complex plane


def screen_to_complex(x: int, y: int) -> complex:
    """Map pixel coordinates to complex plane z = x + iy."""
    re = (x - CENTER_X) / SCALE
    im = -(y - CENTER_Y) / SCALE  # Invert Y axis for standard complex plane
    return complex(re, im)


def complex_to_screen(z: complex) -> tuple[int, int]:
    """Map complex coordinate z = x + iy to screen pixel coordinates."""
    x = int(CENTER_X + z.real * SCALE)
    y = int(CENTER_Y - z.imag * SCALE)
    return x, y


class Particle:
    """Represents a point vortex pole in 2D fluid flow whose motion is

    governed by the complex potentials of surrounding vortices.
    """

    def __init__(self, z: complex, residue: complex):
        self.z = z
        self.residue = residue  # Circulation gamma
        self.radius = 8

    def compute_vortex_velocity(self, all_particles: list["Particle"]) -> complex:
        v_induced = 0.0 + 0.0j
        for other in all_particles:
            if other is self:
                continue
            diff = self.z - other.z
            dist = abs(diff)
            if dist > 0.1:  # Softening core to prevent infinite velocity
                v_induced += (1j / (2 * np.pi)) * (other.residue / diff.conjugate())
        return v_induced

    def update(
        self,
        all_particles: list["Particle"],
        bounds: tuple[float, float, float, float],
        dt: float,
    ):
        v = self.compute_vortex_velocity(all_particles)
        self.z += v * dt

        # Boundary box confinement
        min_re, max_re, min_im, max_im = bounds
        padding = 0.2
        if self.z.real < min_re + padding or self.z.real > max_re - padding:
            self.z = complex(
                np.clip(self.z.real, min_re + padding, max_re - padding),
                self.z.imag,
            )
        if self.z.imag < min_im + padding or self.z.imag > max_im - padding:
            self.z = complex(
                self.z.real,
                np.clip(self.z.imag, min_im + padding, max_im - padding),
            )


class ContourPolygon:
    """Represents a custom freehand closed contour C in the complex plane."""

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
        """Ray-casting algorithm to check if point z lies inside polygon C."""
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
                            xinters = (y - p1.imag) * (p2.real - p1.real) / (
                                p2.imag - p1.imag
                            ) + p1.real
                        if p1.real == p2.real or x <= xinters:
                            inside = not inside
            p1 = p2

        return inside


# --- Setup Simulation Objects ---
particles = [
    Particle(complex(-1.5, 1.2), residue=1.0 + 0.0j),
    Particle(complex(1.2, -1.0), residue=0.5 + 0.5j),
    Particle(complex(0.2, 0.5), residue=-1.0 + 0.0j),
]

contour = ContourPolygon()
font = pygame.font.SysFont("Consolas", 15)

# Initialize a default circular contour loop
default_center = 0 + 0j
default_radius = 2.2
for angle in np.linspace(0, 2 * np.pi, 30, endpoint=False):
    contour.add_point(
        default_center + default_radius * np.exp(1j * angle)
    )
contour.close()

# --- Main Simulation Loop ---
running = True
drawing_contour = False

while running:
    dt = clock.tick(60) / 1000.0  # Delta time in seconds
    screen.fill(BG_COLOR)

    # Complex plane bounds based on window dimensions
    bounds = (-CENTER_X / SCALE, CENTER_X / SCALE, -CENTER_Y / SCALE, CENTER_Y / SCALE)

    # --- Event Handling ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_z = screen_to_complex(*event.pos)

            if event.button == 1:  # Left Click: Spawn Pole Particle
                rand_res = complex(
                    np.random.choice([-1.0, 1.0, 0.5]),
                    np.random.choice([0.0, 0.5, -0.5]),
                )
                particles.append(Particle(mouse_z, rand_res))

            elif event.button == 3:  # Right Click: Start Drawing Custom Contour
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

    # --- Physics & Residue Evaluation ---
    sum_residues = 0.0 + 0.0j
    inside_count = 0

    for p in particles:
        p.update(particles, bounds, dt)
        if contour.contains(p.z):
            sum_residues += p.residue
            inside_count += 1

    # Apply Cauchy's Residue Theorem
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
            for p in particles:
                diff = gz - p.z
                if abs(diff) > 0.15:
                    v_z += p.residue / diff

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

    # 4. Particle Poles
    for p in particles:
        px, py = complex_to_screen(p.z)
        is_inside = contour.contains(p.z)
        color = INSIDE_PARTICLE if is_inside else OUTSIDE_PARTICLE

        pygame.draw.circle(screen, color, (px, py), p.radius)

        res_text = f"{p.residue.real:+.1f}{p.residue.imag:+.1f}i"
        lbl = font.render(res_text, True, (170, 180, 200))
        screen.blit(lbl, (px + 10, py - 10))

    # 5. HUD Dashboard
    hud_data = [
        f"Total Poles: {len(particles)}  |  Inside Contour C: {inside_count}",
        f"Sum of Residues (∑ Res): {sum_residues.real:+.2f} {sum_residues.imag:+.2f}i",
        f"Contour Integral (2πi * ∑ Res): {contour_integral.real:+.2f} {contour_integral.imag:+.2f}i",
        "[Left-Click] Add Vortex Pole  |  [Right-Click + Drag] Draw Closed Contour",
    ]

    for idx, text_str in enumerate(hud_data):
        col = (100, 210, 255) if idx == 2 else (TEXT_COLOR if idx < 3 else (140, 150, 170))
        txt_surface = font.render(text_str, True, col)
        screen.blit(txt_surface, (20, 20 + idx * 24))

    pygame.display.flip()

pygame.quit()
