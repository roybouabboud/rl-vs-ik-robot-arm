"""A Gymnasium environment: move the UR5e's hand to a random target point.

Step 5 of the roadmap (phase 2). Read rl/05_reach_env.md first,
then fill in the TODOs. Check your work with: python rl/check_reach_env.py
"""

import sys
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kinematics"))
from fk import EE_SITE  # noqa: E402
from go_to_point import build_model  # noqa: E402  (UR5e scene + red mocap target sphere)


class ReachEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, max_steps: int = 100, frame_skip: int = 10, action_scale: float = 0.05,
                 success_tol: float = 0.02, render_mode: str | None = None):
        self.model = build_model()
        self.data = mujoco.MjData(self.model)
        self.scratch = mujoco.MjData(self.model)  # for "what-if" FK when sampling targets
        self.site_id = self.model.site(EE_SITE).id
        self.home_id = self.model.key("home").id
        self.ctrl_low, self.ctrl_high = self.model.actuator_ctrlrange.T.copy()

        self.max_steps = max_steps
        self.frame_skip = frame_skip
        self.action_scale = action_scale
        self.success_tol = success_tol
        self.render_mode = render_mode
        self.viewer = None

        self.action_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(21,), dtype=np.float32)

        self.target = np.zeros(3)
        self.steps = 0

    def _get_obs(self) -> np.ndarray:
        """[joint angles (6), joint velocities (6), hand pos (3), target (3), target - hand (3)] as float32."""
        # TODO 1
        raise NotImplementedError

    def _sample_target(self) -> np.ndarray:
        """A random reachable point: FK of random joint angles (on self.scratch), at least 5 cm above
        the floor and at least 20 cm from the base's vertical axis. Use self.np_random."""
        # TODO 2
        raise NotImplementedError

    def _reward(self, distance: float, action: np.ndarray) -> tuple[float, bool]:
        """reward = -distance - 0.01 * |action|^2 + (1 if success). Returns (reward, success)."""
        # TODO 3
        raise NotImplementedError

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)  # seeds self.np_random
        # TODO 4
        raise NotImplementedError

    def step(self, action):
        # TODO 5
        raise NotImplementedError

    # --- viewer (provided) ---------------------------------------------------------
    def render(self):
        if self.render_mode != "human":
            return
        if self.viewer is None:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


if __name__ == "__main__":
    env = ReachEnv()
    obs, _ = env.reset(seed=0)
    print("obs shape", obs.shape, "target", env.target)
    total = 0.0
    for _ in range(env.max_steps):
        obs, r, terminated, truncated, info = env.step(env.action_space.sample())
        total += r
        if terminated or truncated:
            break
    print(f"random episode: return {total:.1f}, final distance {info['distance']:.3f} m")
