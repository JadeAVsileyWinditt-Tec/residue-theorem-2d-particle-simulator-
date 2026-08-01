import numpy as np


class ResidueEngine:
    """
    Complex analysis engine to evaluate pole structures, calculate residues,
    and compute contour integrals along boundary layer paths.
    """
    def __init__(self, poles=None, residues=None):
        """
        :param poles: List of complex numbers representing pole locations.
        :param residues: List of complex numbers representing residues at each pole.
        """
        self.poles = poles if poles is not None else [-0.05 + 0.08j, -0.05 - 0.08j]
        self.residues = residues if residues is not None else [1.0 + 0.0j, -1.0 + 0.0j]

    def evaluate_field(self, z):
        """Evaluates non-analytical field f(z) = sum( R_k / (z - z_k) )."""
        f_val = 0.0 + 0.0j
        for p, r in zip(self.poles, self.residues):
            f_val += r / (z - p)
        return f_val

    def contour_integral_numerical(self, radius=1.0, center=0.0 + 0.0j, num_points=1000):
        """
        Computes the contour integral numerically along a circular loop C:
        I = \oint_C f(z) dz
        """
        theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        dt = theta[1] - theta[0]
        
        # Circular contour parameterization: z(t) = center + radius * exp(i * t)
        z = center + radius * np.exp(1j * theta)
        dz_dt = 1j * radius * np.exp(1j * theta)
        
        integrand = self.evaluate_field(z) * dz_dt
        integral = np.sum(integrand) * dt
        return integral

    def Cauchy_residue_sum(self, radius=1.0, center=0.0 + 0.0j):
        """
        Calculates exact sum via Cauchy's Residue Theorem:
        I = 2 * pi * i * sum(Residues inside contour)
        """
        total_residue = 0.0 + 0.0j
        for p, r in zip(self.poles, self.residues):
            if np.abs(p - center) < radius:
                total_residue += r
        return 2 * np.pi * 1j * total_residue


if __name__ == "__main__":
    engine = ResidueEngine()
    
    # Test circular path enclosing poles
    radius = 0.5
    num_integral = engine.contour_integral_numerical(radius=radius)
    exact_integral = engine.Cauchy_residue_sum(radius=radius)
    
    print(f"--- Residue Calculus Verification ---")
    print(f"Numerical Integral : {num_integral:.6f}")
    print(f"Cauchy Exact       : {exact_integral:.6f}")
    print(f"Absolute Difference: {np.abs(num_integral - exact_integral):.2e}")
