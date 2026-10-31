"""Benchmark the trained SAC policy against the IK solver on the same targets.

Step 7 of the roadmap (phase 2). Read rl/07_benchmark.md, fill in the TODOs, check with
python rl/check_benchmark.py, then:
    python rl/benchmark.py --model checkpoints/sac_outwards_500k/final.zip --episodes 200
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_policy import NormalizedPolicy, RandomPolicy  # noqa: E402
from reach_env import ReachEnv  # noqa: E402  (also puts kinematics/ on sys.path)
from ik import ik_dls  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


class IKPolicy:
    """The step 3 IK solver as a policy with the same predict() interface as an SB3 model.

    When the env's target changes, solve IK once for the goal joint angles. Every step, return the
    action that moves the servo targets (env.data.ctrl) towards that goal, as far as one action
    allows. So IK gets exactly the same speed limit as SAC: |action| <= 1, scaled by env.action_scale."""

    def __init__(self, env: ReachEnv):
        self.env = env
        self.target = None
        self.q_goal = None
        self.solved = False  # did the last IK solve converge?

    def predict(self, obs, deterministic=True):
        if self.target is None or not np.allclose(self.env.target, self.target):
            q_goal, success, _ = ik_dls(self.env.model, self.env.scratch, self.env.target, self.env.data.qpos[:6].copy())
            self.target = self.env.target.copy()
            self.q_goal = q_goal.copy()
            self.solved = success
        action = np.clip((self.q_goal - self.env.data.ctrl) / self.env.action_scale, -1, 1)
        return action, None


def run_episode(policy, env: ReachEnv, seed: int) -> dict:
    """Play one episode (env.reset(seed=seed), deterministic actions) and return:

    success         bool, from the last step's info
    final_distance  hand-target distance at the end [m]
    steps           steps taken
    path_ratio      length of the path the hand travelled / straight-line start-to-target distance
                    (about 1 = straight; a bit under 1 is normal, since the episode ends 2 cm short of the target)
    mean_joint_speed  mean of |qvel| (norm over the 6 joints) across steps [rad/s], lower = calmer
    max_overshoot   largest info["overshoot"] seen [m] (0 if it never went past the target)
    """
    obs, _ = env.reset(seed=seed)
    ee_start = env.data.site_xpos[env.ee_id].copy()
    dist_start = np.linalg.norm(env.target - ee_start)
    path = 0.0
    speeds = []
    max_over = 0.0
    steps = 0
    prev_pos = ee_start

    while True:
        action, _ = policy.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        path += np.linalg.norm(env.data.site_xpos[env.ee_id].copy() - prev_pos)
        speeds.append(np.linalg.norm(env.data.qvel[:6]))
        if info["overshoot"] > max_over: max_over = info["overshoot"]
        prev_pos = env.data.site_xpos[env.ee_id].copy()
        steps += 1
        if terminated or truncated: break

    return {"success": info["is_success"],
            "final_distance": info["distance"],
            "steps": steps,
            "path_ratio": (path / dist_start),
            "mean_joint_speed": np.mean(speeds),
            "max_overshoot": max_over}


def benchmark(policy, env: ReachEnv, n_episodes: int, seed: int = 20_000) -> list[dict]:
    """run_episode on episodes seed, seed + 1, ...: the same targets for every policy."""
    return [run_episode(policy, env, seed + i) for i in range(n_episodes)]


def summarize(results: list[dict]) -> dict:
    """Means over episodes. steps is averaged over successful episodes only (time to reach)."""
    ok = [r for r in results if r["success"]]
    return {
        "success_rate": np.mean([r["success"] for r in results]),
        "final_distance_cm": 100 * np.mean([r["final_distance"] for r in results]),
        "steps_to_reach": np.mean([r["steps"] for r in ok]) if ok else float("nan"),
        "path_ratio": np.mean([r["path_ratio"] for r in results]),
        "joint_speed": np.mean([r["mean_joint_speed"] for r in results]),
        "max_overshoot_cm": 100 * np.mean([r["max_overshoot"] for r in results]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="SAC .zip (uses vecnormalize.pkl next to it if present)")
    parser.add_argument("--episodes", type=int, default=200)
    args = parser.parse_args()

    env = ReachEnv()
    sac = SAC.load(args.model)
    stats = Path(args.model).parent / "vecnormalize.pkl"
    if stats.exists():
        sac = NormalizedPolicy(sac, stats)

    runs = {name: benchmark(p, env, args.episodes)
            for name, p in [("random", RandomPolicy(env)), ("IK", IKPolicy(env)), ("SAC", sac)]}

    rows = {name: summarize(r) for name, r in runs.items()}
    cols = list(rows["IK"])
    print(f"{'':8s}" + "".join(f"{c:>18s}" for c in cols))
    for name, row in rows.items():
        print(f"{name:8s}" + "".join(f"{row[c]:18.3f}" for c in cols))

    ik_ok = np.array([r["success"] for r in runs["IK"]])
    sac_ok = np.array([r["success"] for r in runs["SAC"]])
    print(f"\nboth reach {np.sum(ik_ok & sac_ok)}, only IK {np.sum(ik_ok & ~sac_ok)}, "
          f"only SAC {np.sum(~ik_ok & sac_ok)}, neither {np.sum(~ik_ok & ~sac_ok)}  (of {args.episodes})")
    env.close()
