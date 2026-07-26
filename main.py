import pygame
import cmath
import numpy as np

# Pygame initialization
pygame.init()
WIDTH, HEIGHT = 900, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Residue Theorem 2D Particle Simulator")
clock = pygame.time.Clock()

# Colors
BG_COLOR = (15, 15, 25)
PARTICLE_COLOR = (0, 255, 200)
POLE_COLOR = (255, 70, 70)
TEXT_COLOR = (220, 220, 220)

def screen_to_complex(pos):
    """Convert screen coordinates to complex plane (center at 0,0)."""
    x = (pos[0] - WIDTH / 2) / 100.0
    y = -(pos[1] - HEIGHT / 2) / 100.0
    return complex(x, y)

def complex_to_screen(z):
    """Convert complex plane coordinate to screen pixel location."""
    px = int(WIDTH / 2 + z.real * 100.0)
    py = int(HEIGHT / 2 - z.imag * 100.0)
    return px, py

class Particle:
    def __init__(self, z):
        self.z = z
        self.v = complex(0.0, 0.0)
        self.history = []

    def update(self, force, dt=0.016):
        self.v += force * dt
        self.z += self.v * dt
        
        # Keep trail history
        screen_pos = complex_to_screen(self.z)
        self.history.append(screen_pos)
        if len(self.history) > 40:
            self.history.pop(0)

class Pole:
    def __init__(self, z, strength=complex(-0.5, 1.2)):
        self.z = z
        self.strength = strength  # Real = source/sink, Imag = vortex
        self.radius = 12

class SimulationGUI:
    def __init__(self):
        self.poles = [
            Pole(complex(-1.5, 0.5), complex(-0.8, 2.0)),
            Pole(complex(1.5, -0.5), complex(0.8, -2.0))
        ]
        self.particles = [Particle(complex(np.random.uniform(-3, 3), np.random.uniform(-3, 3))) for _ in range(30)]
        self.dragged_pole = None
        self.potential_flow = False
        self.font = pygame.font.SysFont("Consolas", 16)

    def calculate_force(self, z):
        """Rational function F(z) = sum( strength_k / (z - pole_k) )."""
        if self.potential_flow:
            # Flow past cylinder potential derivative conjugate
            U = 1.0
            R = 0.8
            dw_dz = U * (1.0 - (R**2) / (z**2 if z != 0 else 1e-6))
            return dw_dz.conjugate()
        
        force = complex(0.0, 0.0)
        for pole in self.poles:
            diff = z - pole.z
            if abs(diff) < 0.1:
                diff = cmath.rect(0.1, cmath.phase(diff))
            force += pole.strength / diff
        return force

    def run(self):
        running = True
        while running:
            dt = clock.tick(60) / 1000.0
            screen.fill(BG_COLOR)

            # Process Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_v:
                        self.potential_flow = not self.potential_flow
                    elif event.key == pygame.K_r:
                        self.particles = [Particle(complex(np.random.uniform(-3, 3), np.random.uniform(-3, 3))) for _ in range(30)]
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    m_pos = pygame.mouse.get_pos()
                    for pole in self.poles:
                        px, py = complex_to_screen(pole.z)
                        if (m_pos[0] - px)**2 + (m_pos[1] - py)**2 <= pole.radius**2:
                            self.dragged_pole = pole
                            break
                elif event.type == pygame.MOUSEBUTTONUP:
                    self.dragged_pole = None
                elif event.type == pygame.MOUSEMOTION and self.dragged_pole:
                    self.dragged_pole.z = screen_to_complex(event.pos)

            # Update and Draw Particles
            for p in self.particles:
                f = self.calculate_force(p.z)
                p.update(f, dt)
                
                # Draw trail
                if len(p.history) > 1:
                    pygame.draw.lines(screen, (0, 100, 100), False, p.history, 1)
                
                # Draw particle
                px, py = complex_to_screen(p.z)
                if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                    pygame.draw.circle(screen, PARTICLE_COLOR, (px, py), 4)

            # Draw Poles
            if not self.potential_flow:
                for pole in self.poles:
                    px, py = complex_to_screen(pole.z)
                    pygame.draw.circle(screen, POLE_COLOR, (px, py), pole.radius)
                    pygame.draw.circle(screen, (255, 255, 255), (px, py), pole.radius, 2)

            # Render Overlay Info
            mode_text = "Potential Flow (Cylinder)" if self.potential_flow else "Rational Poles Field"
            txt_surface = self.font.render(f"Mode: {mode_text} | Press [V] Toggle | [R] Reset | Drag Poles", True, TEXT_COLOR)
            screen.blit(txt_surface, (15, 15))

            pygame.display.flip()

        pygame.quit()

if __name__ == "__main__":
    gui = SimulationGUI()
    gui.run()
