# --- Add Vector Field Visualization ---
# Grid setup (compute once outside main loop or keep low res for speed)
GRID_STEP = 30  # Pixel spacing between field arrows
for x in range(0, WIDTH, GRID_STEP):
    for y in range(0, HEIGHT, GRID_STEP):
        z = screen_to_complex(x, y)
        
        # Calculate complex velocity field V(z) = sum(Res_k / (z - z_k))
        v_z = 0.0 + 0.0j
        for p in particles:
            diff = z - p.z
            if abs(diff) > 0.15:  # Avoid singularity division by zero
                v_z += p.residue / diff
        
        # Normalize and scale arrow length
        mag = abs(v_z)
        if mag > 1e-4:
            # Complex conjugation flips imaginary component back to screen space
            direction = (v_z / mag) * min(mag * 5.0, 15.0) 
            end_x = x + int(direction.real)
            end_y = y - int(direction.imag)  # Screen Y inverted
            
            # Fade line color based on field strength
            alpha_col = min(int(mag * 50), 120)
            color = (0, alpha_col + 50, alpha_col + 100)
            pygame.draw.line(screen, color, (x, y), (end_x, end_y), 1)
