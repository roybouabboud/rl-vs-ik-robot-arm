"""Watch a policy that knows nothing: random actions in the ReachEnv viewer.

Run from the repo root:  python rl/watch_random.py
Needs step 5's TODOs done. Close the viewer window to stop.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reach_env import ReachEnv  # noqa: E402

env = ReachEnv(render_mode="human")
returns = []
for episode in range(5):
    obs, _ = env.reset(seed=episode)
    total, info = 0.0, {}
    for _ in range(env.max_steps):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total += reward
        env.render()
        time.sleep(env.frame_skip * env.model.opt.timestep)  # real time
        if terminated or truncated:
            break
    returns.append(total)
    print(f"episode {episode}: return {total:7.1f}, final distance {info['distance'] * 100:5.1f} cm, "
          f"success {info['is_success']}")
print(f"random policy: mean return {np.mean(returns):.1f}")
env.close()
