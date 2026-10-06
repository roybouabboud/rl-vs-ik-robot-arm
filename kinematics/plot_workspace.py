"""3-D scatter of the UR5e's reachable workspace, built from your sample_workspace().

Run from the repo root:  python kinematics/plot_workspace.py
Drag in the plot window to rotate it.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from fk import load_ur5e, sample_workspace  # noqa: E402

N = 10000
SHOULDER = np.array([0.0, 0.0, 0.163])  # shoulder joint position in the world (from ur5e.xml)

model, data = load_ur5e()

points = sample_workspace(model, n=N, seed=0)

# TODO 2: compute each point's distance from SHOULDER, shape (N,).
# Hint: subtract, then np.linalg.norm along the right axis.
dist = np.linalg.norm(points - SHOULDER, axis=1)

print(f"{N} samples, reach from shoulder: min {dist.min():.3f} m, max {dist.max():.3f} m")

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(projection="3d")
sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=dist, cmap="viridis", s=2, alpha=0.5)
ax.scatter(*SHOULDER, color="red", s=80, label="shoulder")
fig.colorbar(sc, ax=ax, shrink=0.6, label="distance from shoulder [m]")

ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_zlabel("z [m]")
ax.set_box_aspect((1, 1, 1))  # equal scale on all axes, so the shape isn't squashed
lim = 1.1
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_zlim(-lim + SHOULDER[2], lim + SHOULDER[2])
ax.set_title("UR5e reachable workspace")
ax.legend()
plt.show()
