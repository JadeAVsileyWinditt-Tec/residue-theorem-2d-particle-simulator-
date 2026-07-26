import numpy as np


class MorseSaddleTopology:
    """
    Simulates Morse-theoretic sublevel sets M_a = {(x,y) | x^2 - y^2 <= a}
    and tracks Betti number transitions (\beta_0, \beta_1) across the critical value.
    """
    def __init__(self, x_bounds=(-2, 2), y_bounds=(-2, 2), resolution=200):
        self.x = np.linspace(x_bounds[0], x_bounds[1], resolution)
        self.y = np.linspace(y_bounds[0], y_bounds[1], resolution)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.S = self.X**2 - self.Y**2  # Morse potential S(x, y) = x^2 - y^2

    def compute_betti_numbers(self, a):
        """
        Determines theoretical Betti numbers (\beta_0, \beta_1) for sublevel set M_a.
        
        - a < 0: Two disconnected unbounded components (\beta_0 = 2, \beta_1 = 0)
        - a = 0: Critical threshold (Components touch at origin)
        - a > 0: Single connected component with non-contractible loop (\beta_0 = 1, \beta_1 = 1)
        """
        if a < 0:
            return {'beta_0': 2, 'beta_1': 0, 'state': 'Disconnected components'}
        elif a == 0:
            return {'beta_0': 1, 'beta_1': 0, 'state': 'Critical Saddle Horizon'}
        else:
            return {'beta_0': 1, 'beta_1': 1, 'state': 'Connected trapped cycle'}

    def evaluate_sublevel_mask(self, a):
        """Returns boolean grid where S(x, y) <= a."""
        return self.S <= a


if __name__ == "__main__":
    morse = MorseSaddleTopology()
    
    thresholds = [-0.5, 0.0, 0.5]
    print("--- Morse Topological Phase Transition Analysis ---")
    for a in thresholds:
        betti = morse.compute_betti_numbers(a)
        print(f"Threshold a = {a:+0.1f} | State: {betti['state']:<25} | Betti Numbers: beta_0 = {betti['beta_0']}, beta_1 = {betti['beta_1']}")
