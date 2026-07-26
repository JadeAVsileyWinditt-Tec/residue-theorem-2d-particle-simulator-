class Particle:
    """Represents a point vortex in 2D fluid flow whose motion is governed

    by the complex potential of surrounding vortices.
    """

    def __init__(self, z: complex, residue: complex):
        self.z = z
        self.residue = residue  # Circulation gamma
        self.radius = 8

    def compute_vortex_velocity(self, all_particles: list['Particle']) -> complex:
        # Sum velocity induced by all other vortices: dz_k/dt = (-i / 2pi) * sum(Res_j / (z_k - z_j)^*)
        v_induced = 0.0 + 0.0j
        for other in all_particles:
            if other is self:
                continue
            diff = self.z - other.z
            dist = abs(diff)
            if dist > 0.1:  # Softening core to avoid infinite velocity
                # Complex conjugate of 1/(z_k - z_j)
                v_induced += (1j / (2 * np.pi)) * (other.residue / diff.conjugate())

        return v_induced

    def update(
        self,
        all_particles: list['Particle'],
        bounds: tuple[float, float, float, float],
        dt: float,
    ):
        v = self.compute_vortex_velocity(all_particles)
        self.z += v * dt

        # Soft boundary box confinement
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
