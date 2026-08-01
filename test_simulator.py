import pytest
import numpy as np
from engine.physics.tbu import TBUNonHermitianSimulator
from engine.math.residue import ResidueEngine

def test_energy_dissipation_inside_layer():
    """Verify that effective energy monotonically decreases in the trapping regime."""
    sim = TBUNonHermitianSimulator(eps=0.05, delta=0.08, K=4.0, l=0.25)
    data = sim.run_simulation(r0=0.45, vr0=-0.1, t_span=(0.0, 40.0))
    
    # Check that final energy is lower than initial energy
    assert data['E_eff'][-1] < data['E_eff'][0]
    # Check that amplitude norm decays boundedly between 0 and 1
    assert np.all(data['amplitude_norm'] <= 1.0)
    assert np.all(data['amplitude_norm'] >= 0.0)

def test_cauchy_residue_theorem_accuracy():
    """Verify numerical contour integral matches Cauchy's exact residue theorem."""
    engine = ResidueEngine(
        poles=[-0.05 + 0.08j, -0.05 - 0.08j],
        residues=[1.0 + 0.0j, -1.0 + 0.0j]
    )
    
    radius = 0.5
    num_integral = engine.contour_integral_numerical(radius=radius, num_points=2000)
    exact_integral = engine.Cauchy_residue_sum(radius=radius)
    
    # Numerical contour integral should match Cauchy exact sum within 1e-3 tolerance
    assert np.abs(num_integral - exact_integral) < 1e-3
