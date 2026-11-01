"""A Gymnasium environment: move the UR5e's hand to a random target point."""

import sys
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kinematics"))
from fk import EE_SITE, ee_pose, sample_workspace  # noqa: E402
from go_to_point import build_model  # noqa: E402  (UR5e scene + red mocap target sphere)
BASE_BODY = "base"


class ReachEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, max_steps: int = 200, frame_skip: int = 10, action_scale: float = 0.025,
                 success_tol: float = 0.02, render_mode: str | None = None, sigma_near: float = 0.1, 
                 sigma_fine: float = 0.02, fine_weight: float = 0.5,
                 bonus: float = 10, overshoot_weight: float = 0.0, action_weight: float = 0.01, 
                 hold_steps=10, stayin_weight=0.5, rate_weight=0.05):
        self.model = build_model()
        self.data = mujoco.MjData(self.model)
        self.scratch = mujoco.MjData(self.model)  # for "what-if" FK when sampling targets
        self.ee_id = self.model.site(EE_SITE).id
        self.base_id = self.model.body(BASE_BODY).id
        self.home_id = self.model.key("home").id
        self.ctrl_low, self.ctrl_high = self.model.actuator_ctrlrange.T.copy()

        self.max_steps = max_steps
        self.frame_skip = frame_skip
        self.action_scale = action_scale
        self.success_tol = success_tol
        self.render_mode = render_mode
        self.viewer = None

        self.sigma_near = sigma_near
        self.sigma_fine = sigma_fine
        self.fine_weight = fine_weight
        self.bonus = bonus
        self.overshoot_weight = overshoot_weight
        self.action_weight = action_weight
        self.stayin_weight = stayin_weight
        self.rate_weight = rate_weight

        self.in_tol_steps = 0
        self.prev_action = np.zeros(6)
        self.hold_steps = hold_steps

        self.action_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(27,), dtype=np.float32)

        self.target = np.zeros(3)
        self.steps = 0

    def _get_obs(self) -> np.ndarray:
        """[joint angles (6), joint velocities (6), hand pos (3), target (3), target - hand (3), previous action (6)] as float32."""
        q = self.data.qpos[:]
        dq = self.data.qvel[:]
        pos, _ = ee_pose(self.model, self.data, q)
        target = self.target
        dist = target - pos
        res = np.concatenate([q, dq, pos, target, dist, self.prev_action], 0)
        return res.astype(np.float32)


    def _sample_target(self) -> np.ndarray:
        """A random reachable point: FK of random joint angles (on self.scratch), at least 5 cm above
        the floor and at least 20 cm from the base's vertical axis, from a pose where the arm doesn't touch
        the floor or itself (so at least one collision-free way to reach it exists). Use self.np_random."""
        lims = self.model.jnt_range
        low_lims = lims[:,0]
        high_lims = lims[:, 1]
        while True:
            q_rand = self.np_random.uniform(low_lims, high_lims)
            point, _ = ee_pose(self.model, self.scratch, q_rand)
            if point[2] > 0.05 and np.linalg.norm(point[:2]) > 0.2 and not self._in_collision(q_rand):
                return point

    def _in_collision(self, q: np.ndarray) -> bool:
        """True if the arm at joint angles q touches the floor or itself (checked on self.scratch)."""
        self.scratch.qpos[:6] = q
        mujoco.mj_forward(self.model, self.scratch)
        return self.scratch.ncon > 0
        

    def _reward(self, distance: float, action: np.ndarray, overshoot: float=0.0, in_tol_steps: int=0,
                action_pen: float=0) -> tuple[float, bool]:
        """reward = -distance - action_weight * |action|^2 + (exp(-d/sigma_near) - 1)
        + fine_weight * (exp(-d/sigma_fine) - 1) - overshoot_weight * max(0, overshoot)
        + stayin_weight * in_tol_steps - rate_weight * |action - prev_action|^2 + (bonus once held for hold_steps). Returns (reward, success)."""
        success = False
        reward = np.float32(-1*distance - self.action_weight*np.linalg.norm(action)**2 + (np.exp(-1*distance/self.sigma_near)-1)
                            + self.fine_weight*(np.exp(-1*distance/self.sigma_fine)-1) 
                            - self.overshoot_weight*max(0, overshoot) + self.stayin_weight*in_tol_steps - self.rate_weight*action_pen)
        if in_tol_steps >= self.hold_steps: 
            reward += self.bonus
            success = True
        return reward, success


    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)  # seeds self.np_random
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_id)
        q_noise = self.np_random.uniform(-0.1, 0.1, size=6)
        self.data.qpos[:6] += q_noise
        self.data.ctrl = self.data.qpos[:6]
        self.target = self._sample_target()
        self.data.mocap_pos[0] = self.target
        mujoco.mj_forward(self.model, self.data)
        self.in_tol_steps = 0
        self.prev_action = np.zeros(6)
        self.steps = 0
        return self._get_obs(), {}


    def step(self, action):
        act = np.clip(action, -1, 1)
        self.data.ctrl += act * self.action_scale
        self.data.ctrl = np.clip(self.data.ctrl, self.ctrl_low, self.ctrl_high)
        for i in range(self.frame_skip): mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)  # mj_step leaves site_xpos one step stale

        # Distance from EE to target + # of consecutive times within target 
        dist = np.linalg.norm(self.target - self.data.site_xpos[self.ee_id])
        if dist <= self.success_tol: self.in_tol_steps += 1
        else: self.in_tol_steps = 0

        # Outwards (base -> target) Approach
        u = self.target - self.data.xpos[self.base_id]
        u[2] = 0
        overshoot = 0
        if np.linalg.norm(u) > 1e-6:
            v = self.data.site_xpos[self.ee_id] - self.data.xpos[self.base_id]
            overshoot = (np.dot(u, v) / np.linalg.norm(u)) - np.linalg.norm(u)

        # Action Rate Penalty
        action_pen = np.linalg.norm(act - self.prev_action)**2
        self.prev_action = act.copy()

        reward, success = self._reward(dist, act, overshoot, self.in_tol_steps, action_pen)
        self.steps += 1
        return self._get_obs(), reward, success, self.steps >= self.max_steps, {"distance": dist, "is_success": success, "overshoot": overshoot, "in_tol_steps": self.in_tol_steps, "action_change": action_pen}


    # --- viewer -------------------------------------------------------------------
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
