"""Plot how well-conditioned the UR5e's Jacobian is as the elbow and wrist_2 joints sweep.

Run from the repo root:  python kinematics/explore_singularity.py
Needs TODOs 1 and 4 in jacobian.py done.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from fk import load_ur5e  # noqa: E402
from jacobian import jacobian_condition  # noqa: E402

model, data = load_ur5e()
BASE_Q = np.array([0.0, -1.0, 1.0, -1.0, 1.0, 0.0])  # a comfortable, bent pose
SWEEPS = [(2, "elbow"), (4, "wrist_2")]
angles = np.linspace(-np.pi, np.pi, 721)

fig, axes = plt.subplots(1, len(SWEEPS), figsize=(12, 4.5), sharey=True)
for ax, (joint, name) in zip(axes, SWEEPS):
    for full, label in [(True, "full 6x6"), (False, "position-only 3x6")]:
        conds = []
        for a in angles:
            q = BASE_Q.copy()
            q[joint] = a
            conds.append(jacobian_condition(model, data, q, full=full))
        ax.semilogy(angles, np.clip(conds, 1, 1e8), label=label)
    ax.set_title(f"sweeping {name} (other joints fixed)")
    ax.set_xlabel(f"{name} angle [rad]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
axes[0].set_ylabel("condition number (log, clipped at 1e8)")
fig.suptitle("Where does the UR5e's Jacobian break down?")
fig.tight_layout()
plt.show()
