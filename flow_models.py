import cmath
import numpy as np
from abc import ABC, abstractmethod


class PotentialFlow(ABC):
    """Abstract base class for complex potential flows w(z) = phi + i*psi."""
    
    @abstractmethod
    def complex_potential(self, z: complex) -> complex:
        """Evaluates w(z) = phi + i*psi."""
        pass

    @abstractmethod
    def complex_velocity(self, z: complex) -> complex:
        """Evaluates w'(z) = u - i*v, so velocity V = conj(w'(z))."""
        pass

    def velocity_vector(self, z: complex) -> tuple[float, float]:
        """Returns physical (u, v) real velocity vector at point z."""
        dw_dz = self.complex_velocity(z)
        V = dw_dz.conjugate()
        return V.real, V.imag


class UniformFlow(PotentialFlow):
    def __init__(self, U: float = 1.0, alpha_deg: float = 0.0):
        self.U = U
        self.alpha = np.radians(alpha_deg)

    def complex_potential(self, z: complex) -> complex:
        return self.U * z * cmath.exp(-1j * self.alpha)

    def complex_velocity(self, z: complex) -> complex:
        return self.U * cmath.exp(-1j * self.alpha)


class CylinderWithCirculation(PotentialFlow):
    """Flow past cylinder with radius R and circulation Gamma (Kutta-Joukowski/Magnus effect)."""
    def __init__(self, U: float = 1.0, R: float = 1.0, gamma: float = 2.0, center: complex = 0.0 + 0.0j):
        self.U = U
        self.R = R
        self.gamma = gamma
        self.center = center

    def complex_potential(self, z: complex) -> complex:
        zc = z - self.center
        if abs(zc) < 1e-12:
            zc = 1e-12 + 0j
        return self.U * (zc + (self.R**2) / zc) + 1j * (self.gamma / (2 * np.pi)) * cmath.log(zc / self.R)

    def complex_velocity(self, z: complex) -> complex:
        zc = z - self.center
        if abs(zc) < 1e-6:
            zc = 1e-6 + 0j
        return self.U * (1.0 - (self.R**2) / (zc**2)) + 1j * (self.gamma / (2 * np.pi * zc))


class JoukowskyAirfoil(PotentialFlow):
    """
    Conformal Mapping Z = z + a^2 / z transforming flow around an offset 
    cylinder into physical potential flow around a lifting airfoil.
    """
    def __init__(self, U: float = 1.0, a: float = 1.0, dx: float = -0.1, dy: float = 0.1, alpha_deg: float = 5.0):
        self.a = a
        self.alpha = np.radians(alpha_deg)
        # Cylinder center offset in zeta-plane
        self.zeta_c = complex(dx, dy)
        # Cylinder radius passing through trailing edge z = a
        self.R = abs(complex(a, 0) - self.zeta_c)
        
        # Enforce Kutta condition at trailing edge
        self.gamma = 4 * np.pi * U * self.R * np.sin(self.alpha + np.arcsin(dy / self.R))
        
        # Base flow in cylinder (zeta) plane
        self.base_flow = CylinderWithCirculation(U=U, R=self.R, gamma=self.gamma, center=self.zeta_c)

    def complex_potential(self, z: complex) -> complex:
        # Inverse mapping Z -> zeta
        disc = z**2 - 4 * (self.a**2)
        zeta = (z + cmath.sqrt(disc)) / 2.0
        return self.base_flow.complex_potential(zeta)

    def complex_velocity(self, z: complex) -> complex:
        # Chain rule: dW/dZ = (dW/dzeta) / (dZ/dzeta)
        disc = z**2 - 4 * (self.a**2)
        zeta = (z + cmath.sqrt(disc)) / 2.0
        
        dW_dzeta = self.base_flow.complex_velocity(zeta)
        dZ_dzeta = 1.0 - (self.a**2) / (zeta**2 if abs(zeta) > 1e-6 else 1e-6)
        
        if abs(dZ_dzeta) < 1e-5:  # Singular trailing edge check
            return 0.0 + 0.0j
            
        return dW_dzeta / dZ_dzeta


class SuperpositionFlow(PotentialFlow):
    """Composite container allowing dynamic composition of arbitrary potential flows."""
    def __init__(self, flows=None):
        self.flows = flows if flows is not None else []

    def add_flow(self, flow: PotentialFlow):
        self.flows.append(flow)

    def complex_potential(self, z: complex) -> complex:
        return sum((f.complex_potential(z) for f in self.flows), 0.0 + 0.0j)

    def complex_velocity(self, z: complex) -> complex:
        return sum((f.complex_velocity(z) for f in self.flows), 0.0 + 0.0j)
