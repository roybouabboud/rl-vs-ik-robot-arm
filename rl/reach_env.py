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
from fk import EE_SITE, ee_pose, sample_workspace  # noqa: E402
from go_to_point import build_model  # noqa: E402  (UR5e scene + red mocap target sphere)


class ReachEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, max_steps: int = 200, frame_skip: int = 10, action_scale: float = 0.025,
                 success_tol: float = 0.02, render_mode: str | None = None, sigma: float = 0.1,
                 bonus: float = 10):
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
        self.sigma = sigma
        self.bonus = bonus

        self.action_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(21,), dtype=np.float32)

        self.target = np.zeros(3)
        self.steps = 0

    def _get_obs(self) -> np.ndarray:
        """[joint angles (6), joint velocities (6), hand pos (3), target (3), target - hand (3)] as float32."""
        q = self.data.qpos[:]
        dq = self.data.qvel[:]
        pos, _ = ee_pose(self.model, self.data, q)
        target = self.target
        dist = target - pos
        res = np.concatenate([q, dq, pos, target, dist], 0)
        return res.astype(np.float32)


    def _sample_target(self) -> np.ndarray:
        """A random reachable point: FK of random joint angles (on self.scratch), at least 5 cm above
        the floor and at least 20 cm from the base's vertical axis. Use self.np_random."""
        lims = self.model.jnt_range
        low_lims = lims[:,0]
        high_lims = lims[:, 1]
        while True:
            q_rand = self.np_random.uniform(low_lims, high_lims)
            point, _ = ee_pose(self.model, self.scratch, q_rand)
            if point[2] > 0.05 and np.linalg.norm(point[:2]) > 0.2:
                return point
        

    def _reward(self, distance: float, action: np.ndarray) -> tuple[float, bool]:
        """reward = -distance - 0.01 * |action|^2 + (exp(-d/sigma) - 1) + (bonus if success). Returns (reward, success)."""
        success = False
        reward = np.float32(-1*distance - 0.01*np.linalg.norm(action)**2 + (np.exp(-1*distance/self.sigma)-1))
        if distance <= self.success_tol: 
            reward += self.bonus
            success = True
        return reward, success
    # TO IMPROVE FOR BETTER CONVERGENCE: reward, tolerance, action size, observation


    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)  # seeds self.np_random
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_id)
        q_noise = self.np_random.uniform(-0.1, 0.1, size=6)
        self.data.qpos[:6] += q_noise
        self.data.ctrl = self.data.qpos[:6]
        self.target = self._sample_target()
        self.data.mocap_pos[0] = self.target
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        return self._get_obs(), {}


    def step(self, action):
        act = np.clip(action, -1, 1)
        self.data.ctrl += act * self.action_scale
        self.data.ctrl = np.clip(self.data.ctrl, self.ctrl_low, self.ctrl_high)
        for i in range(self.frame_skip): mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)  # mj_step leaves site_xpos one step stale
        dist = np.linalg.norm(self.target - self.data.site_xpos[self.site_id])
        reward, success = self._reward(dist, act)
        self.steps += 1
        return self._get_obs(), reward, success, self.steps >= self.max_steps, {"distance": dist, "is_success": success}


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
