import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Parameters ---
GRID_SIZE = 5         # Size of the grid (GRID_SIZE x GRID_SIZE)
ALIVE_PROB = 0.2       # Initial probability of a cell being alive
UPDATE_INTERVAL = 200  # Milliseconds between frames

# --- Initialize grid randomly ---
grid = np.random.choice([0, 1], size=(GRID_SIZE, GRID_SIZE), p=[1-ALIVE_PROB, ALIVE_PROB])

# --- Function to compute next generation ---
def update(frame_num, img, grid):
    new_grid = grid.copy()
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            # Count live neighbors (with periodic boundary)
            total = (
                grid[i, (j-1)%GRID_SIZE] + grid[i, (j+1)%GRID_SIZE] +
                grid[(i-1)%GRID_SIZE, j] + grid[(i+1)%GRID_SIZE, j] +
                grid[(i-1)%GRID_SIZE, (j-1)%GRID_SIZE] + grid[(i-1)%GRID_SIZE, (j+1)%GRID_SIZE] +
                grid[(i+1)%GRID_SIZE, (j-1)%GRID_SIZE] + grid[(i+1)%GRID_SIZE, (j+1)%GRID_SIZE]
            )

            # Apply Conway's rules
            if grid[i, j] == 1:
                if total < 2 or total > 3:
                    new_grid[i, j] = 0  # Dies
            else:
                if total == 3:
                    new_grid[i, j] = 1  # Becomes alive

    img.set_data(new_grid)
    grid[:] = new_grid[:]
    return img

# --- Set up animation ---
fig, ax = plt.subplots()
img = ax.imshow(grid, interpolation='nearest', cmap='binary')
ax.axis('off')

ani = animation.FuncAnimation(fig, update, fargs=(img, grid), interval=UPDATE_INTERVAL)
plt.show()