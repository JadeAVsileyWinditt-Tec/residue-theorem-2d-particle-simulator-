class ContourPolygon:
    """Simple polygon used for drawing and testing closed contours (residue theorem)."""

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
        """Ray-casting point-in-polygon test."""
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
