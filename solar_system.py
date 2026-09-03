Python 3.14.0 (tags/v3.14.0:ebf955d, Oct  7 2025, 10:15:03) [MSC v.1944 64 bit (AMD64)] on win32
Enter "help" below or click "Help" above for more information.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# Constants in chosen units
G = 4 * np.pi**2  # AU^3 / (yr^2 * solar_mass)

# Planet definitions: name, mass (solar masses), initial distance from Sun (AU)
# We'll place bodies on x-axis and give them circular velocities for initial conditions.
bodies = [
    # name      mass(M_sun)      x (AU)       y (AU)     vx (AU/yr) vy (AU/yr) -- set later
    ("Sun",     1.000000,        0.0,         0.0,       0.0,       0.0),
    ("Mercury", 1.660e-7,        0.387,       0.0,       0.0,       0.0),
    ("Venus",   2.447e-6,        0.723,       0.0,       0.0,       0.0),
    ("Earth",   3.003e-6,        1.000,       0.0,       0.0,       0.0),
    ("Mars",    3.213e-7,        1.524,       0.0,       0.0,       0.0),
    ("Jupiter", 9.545e-4,        5.203,       0.0,       0.0,       0.0),
    ("Saturn",  2.858e-4,        9.537,       0.0,       0.0,       0.0),
    # Add Uranus/Neptune if desired:
    ("Uranus",  4.366e-5,       19.191,       0.0,       0.0,       0.0),
    ("Neptune", 5.151e-5,       30.068,       0.0,       0.0,       0.0),
]

N = len(bodies)

# Turn bodies into arrays
names = [b[0] for b in bodies]
masses = np.array([b[1] for b in bodies], dtype=float)

pos = np.zeros((N, 2), dtype=float)  # x,y positions
vel = np.zeros((N, 2), dtype=float)  # vx,vy

for i, b in enumerate(bodies):
    pos[i, 0] = b[2]  # x
    pos[i, 1] = b[3]  # y

# Give initial velocities for roughly circular orbits about the Sun (approx):
# v = sqrt(G * M_sun / r) directed +y if position on +x
for i in range(1, N):
    r = np.linalg.norm(pos[i])
    # circular velocity magnitude (approx, about the Sun only)
    v_circ = np.sqrt(G * masses[0] / r)
    vel[i, 1] = v_circ
# Give Sun an initial zero momentum offset so total momentum is zero (center-of-mass frame)
total_momentum = np.sum(masses[:, None] * vel, axis=0)
vel[0] = - total_momentum / masses[0]

def accelerations(positions):
    acc = np.zeros_like(positions)
    for i in range(N):
        # vector from i to all j
        r_ij = positions - positions[i]
        dist_sq = np.sum(r_ij**2, axis=1)
        # avoid self
        dist_sq[i] = 1.0
        inv_dist3 = dist_sq**(-1.5)
        inv_dist3[i] = 0.0
        # accumulate G * m_j * r_ij * inv_dist3
        acc[i] = G * np.sum((masses[:, None] * r_ij) * inv_dist3[:, None], axis=0)
    return acc

# Simulation parameters
dt = 0.002     # years per step (~0.002 yr ≈ 0.73 days). Reduce for more accuracy.
steps = 20000  # total steps (20000 * 0.002 yr ≈ 40 years). Tune as needed.
plot_interval = 4  # plot every k steps to speed up animation

# For animation/trails, we'll simulate and store selected frames
positions_history = []

# Integrate using velocity-Verlet and store positions periodically
pos_curr = pos.copy()
vel_curr = vel.copy()
acc_curr = accelerations(pos_curr)

for step in range(steps):
    # Velocity-Verlet
    pos_next = pos_curr + vel_curr * dt + 0.5 * acc_curr * dt**2
    acc_next = accelerations(pos_next)
    vel_next = vel_curr + 0.5 * (acc_curr + acc_next) * dt

    pos_curr, vel_curr, acc_curr = pos_next, vel_next, acc_next

    if step % plot_interval == 0:
        positions_history.append(pos_curr.copy())

positions_history = np.array(positions_history)  # shape: (frames, N, 2)
frames = positions_history.shape[0]

# --- Plot / Animation ---
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_facecolor("k")
ax.set_aspect('equal', 'box')

# Determine plot limits based on outermost planet (Neptune)
... max_dist = np.max(np.linalg.norm(positions_history, axis=2))
... lim = max(10.0, max_dist * 1.1)  # show a good region
... ax.set_xlim(-lim, lim)
... ax.set_ylim(-lim, lim)
... ax.set_xlabel("x (AU)")
... ax.set_ylabel("y (AU)")
... ax.set_title("Simplified N-body Solar System (AU, years)")
... 
... # Marker sizes (scale up for visibility) and colors
... sizes = np.array([20, 3, 4, 4, 3, 10, 9, 6, 6]) * 10  # tweak for visibility
... colors = ['yellow', 'gray', 'orange', 'blue', 'red', 'brown', 'gold', 'cyan', 'navy']
... 
... scat = ax.scatter([], [], s=sizes, color=colors)
... # text labels
... labels = [ax.text(0,0,"", color='white', fontsize=8) for _ in range(N)]
... 
... # trails: lines for each body
... trail_len = 200  # points
... lines = [ax.plot([], [], linewidth=1, alpha=0.6, color=colors[i])[0] for i in range(N)]
... 
... def init():
...     scat.set_offsets([])
...     for lbl in labels:
...         lbl.set_text("")
...     for line in lines:
...         line.set_data([], [])
...     return [scat, *labels, *lines]
... 
... def animate(frame):
...     data = positions_history[frame]
...     scat.set_offsets(data)
...     # update labels and trails
...     for i in range(N):
...         x, y = data[i]
...         labels[i].set_position((x, y))
...         labels[i].set_text(names[i])
...         # compute trail points
...         start = max(0, frame - trail_len)
...         trail = positions_history[start:frame+1, i]
...         lines[i].set_data(trail[:,0], trail[:,1])
...     return [scat, *labels, *lines]
... 
... anim = animation.FuncAnimation(fig, animate, frames=frames, init_func=init,
...                                interval=30, blit=True)
... 
... plt.show()
