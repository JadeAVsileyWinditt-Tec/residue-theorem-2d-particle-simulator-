import pygame
import cmath
import numpy as np

from flow_models import UniformFlow, CylinderWithCirculation, JoukowskyAirfoil
from physics_engine import RK4ParticleEngine

# --- Pygame Setup ---
pygame.init()
WIDTH, HEIGHT = 900, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Residue Theorem 2D Particle Simulator - Production Engine")
clock = pygame.time.Clock()

# --- Visual Styling ---
BG_COLOR = (12, 14, 22)
PARTICLE_COLOR = (0, 255, 200)
AIRFOIL_COLOR = (255, 180, 50)
TEXT_COLOR = (220, 230, 240)
GRID_COLOR = (25, 30, 45)


def screen_to_complex(pos):
    """Map screen pixel coordinates to complex z-plane centered at (0, 0)."""
    x = (pos[0] - WIDTH / 2) / 100.0
    y = -(pos[1] - HEIGHT / 2) / 100.0
    return complex(x, y)


def complex_to_screen(z):
    """Map complex z-plane coordinates to screen pixel position."""
    px = int(WIDTH / 2 + z.real * 100.0)
    py = int(HEIGHT / 2 - z.imag * 100.0)
    return px, py


class VisualParticle:
    def __init__(self, z, v=complex(0.0, 0.0)):
        self.z = z
        self.v = v
        self.history = []

    def update_trail(self):
        px, py = complex_to_screen(self.z)
        self.history.append((px, py))
        if len(self.history) > 35:
            self.history.pop(0)


class InteractiveSimulatorApp:
    def __init__(self):
        # Register Available Production Flows
        self.flows = {
            "1": ("Uniform Flow", UniformFlow(U=1.5, alpha_deg=10)),
            "2": ("Magnus Cylinder (Rotating)", CylinderWithCirculation(U=1.2, R=0.8, gamma=3.5)),
            "3": ("Joukowsky Airfoil", JoukowskyAirfoil(U=1.2, a=0.8, dx=-0.08, dy=0.08, alpha_deg=8))
        }
        
        self.active_key = "2"
        self.flow_name, self.current_flow = self.flows[self.active_key]
        self.engine = RK4ParticleEngine(flow_field=self.current_flow, mass=1.0, drag=0.02)
        
        self.particles = []
        self.spawn_particle_grid()
        self.font = pygame.font.SysFont("Consolas", 15)

    def spawn_particle_grid(self):
        """Spawns an upstream column of tracer particles."""
        self.particles = []
        for y_val in np.linspace(-3.0, 3.0, 40):
            z_init = complex(-4.2, y_val)
            v_init = complex(1.2, 0.0)
            self.particles.append(VisualParticle(z=z_init, v=v_init))

    def switch_flow(self, key):
        if key in self.flows:
            self.active_key = key
            self.flow_name, self.current_flow = self.flows[key]
            # Re-bind engine field
            self.engine.field = self.current_flow
            self.spawn_particle_grid()

    def draw_grid(self):
        """Draws subtle background complex coordinate axes."""
        cx, cy = complex_to_screen(0 + 0j)
        pygame.draw.line(screen, GRID_COLOR, (0, cy), (WIDTH, cy), 1)
        pygame.draw.line(screen, GRID_COLOR, (cx, 0), (cx, HEIGHT), 1)

    def draw_obstacles(self):
        """Draws physical boundaries depending on the active potential model."""
        if isinstance(self.current_flow, CylinderWithCirculation):
            # Render Cylinder Boundary
            center = self.current_flow.center
            radius = self.current_flow.R
            cx, cy = complex_to_screen(center)
            r_px = int(radius * 100.0)
            pygame.draw.circle(screen, (40, 50, 70), (cx, cy), r_px)
            pygame.draw.circle(screen, AIRFOIL_COLOR, (cx, cy), r_px, 2)

        elif isinstance(self.current_flow, JoukowskyAirfoil):
            # Render Airfoil Boundary by mapping circle z = zeta_c + R * exp(i*t)
            theta = np.linspace(0, 2 * np.pi, 150)
            circle_points = self.current_flow.zeta_c + self.current_flow.R * np.exp(1j * theta)
            
            # Joukowsky Transform Z = z + a^2 / z
            airfoil_Z = circle_points + (self.current_flow.a**2) / circle_points
            
            screen_pts = [complex_to_screen(z_pt) for z_pt in airfoil_Z]
            if len(screen_pts) > 2:
                pygame.draw.polygon(screen, (40, 50, 70), screen_pts)
                pygame.draw.lines(screen, AIRFOIL_COLOR, True, screen_pts, 2)

    def run(self):
        running = True
        while running:
            dt = clock.tick(60) / 1000.0
            dt = min(dt, 0.033)  # Clamp maximum step time
            
            screen.fill(BG_COLOR)
            self.draw_grid()

            # --- Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_1, pygame.K_KP1):
                        self.switch_flow("1")
                    elif event.key in (pygame.K_2, pygame.K_KP2):
                        self.switch_flow("2")
                    elif event.key in (pygame.K_3, pygame.K_KP3):
                        self.switch_flow("3")
                    elif event.key == pygame.K_r:
                        self.spawn_particle_grid()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Spawn particle directly at clicked location
                    z_click = screen_to_complex(pygame.mouse.get_pos())
                    self.particles.append(VisualParticle(z=z_click))

            # --- Physics & Particle Render Loop ---
            for p in self.particles:
                # Step physics using RK4 Integrator engine
                p.z, p.v = self.engine.step_particle(p.z, p.v, dt)
                p.update_trail()

                # Draw trail lines
                if len(p.history) > 1:
                    pygame.draw.lines(screen, (0, 110, 100), False, p.history, 1)

                # Draw particle head
                px, py = complex_to_screen(p.z)
                if -50 <= px < WIDTH + 50 and -50 <= py < HEIGHT + 50:
                    pygame.draw.circle(screen, PARTICLE_COLOR, (px, py), 3)

            # Draw geometry obstacles
            self.draw_obstacles()

            # --- HUD Overlay ---
            info_lines = [
                f"Active Flow: {self.flow_name}",
                "Controls: Press [1] Uniform | [2] Rotating Cylinder | [3] Airfoil | [R] Reset | [Click] Add Particle"
            ]
            for i, line in enumerate(info_lines):
                txt_surface = self.font.render(line, True, TEXT_COLOR)
                screen.blit(txt_surface, (15, 15 + i * 22))

            pygame.display.flip()

        pygame.quit()


if __name__ == "__main__":
    app = InteractiveSimulatorApp()
    app.run()
