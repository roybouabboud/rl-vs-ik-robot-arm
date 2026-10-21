"""Score a trained policy on ReachEnv, compare it with random actions, and optionally watch it.

Step 6 of the roadmap (phase 2). Fill in the TODO, then:
    python rl/eval_policy.py                    # scores checkpoints/sac_reach/best_model.zip
    python rl/eval_policy.py --render           # ...and shows it in the viewer
    python rl/eval_policy.py --model checkpoints/sac_reach/final.zip
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reach_env import ReachEnv  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


class RandomPolicy:
    """Same predict() interface as an SB3 model, but acts randomly. The baseline to beat."""

    def __init__(self, env):
        self.action_space = env.action_space

    def predict(self, obs, deterministic=True):
        return self.action_space.sample(), None


class NormalizedPolicy:
    """Wraps a model trained with VecNormalize: rescales each raw observation with the saved training
    statistics before asking the model for an action. Same predict() interface, so evaluate() needs no changes."""

    def __init__(self, model, stats_path):
        self.model = model
        self.vec_norm = VecNormalize.load(stats_path, DummyVecEnv([ReachEnv]))
        self.vec_norm.training = False  # use the saved statistics, don't keep updating them

    def predict(self, obs, deterministic=True):
        return self.model.predict(self.vec_norm.normalize_obs(obs), deterministic=deterministic)


def evaluate(model, env: ReachEnv, n_episodes: int = 50, seed: int = 10_000,
             render: bool = False) -> dict:
    """Run n_episodes, resetting episode i with seed + i, using the policy's deterministic action.

    Returns {"success_rate", "mean_final_distance", "mean_return", "mean_steps"}: success_rate in [0, 1],
    final distance in metres (from the last step's info), mean_steps = steps per episode.
    If render: call env.render() each step and sleep so it plays in real time."""
    successes = []
    fin_dist = []
    returns = []
    steps = []

    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i)  # reset returns (obs, info)
        total = 0.0
        ctr = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)  # (action, state); state is unused
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            ctr += 1
            if render:
                env.render()
                time.sleep(env.frame_skip * env.model.opt.timestep)  # real time
            if terminated or truncated:
                break
        successes.append(info["is_success"])
        fin_dist.append(info["distance"])
        returns.append(total)
        steps.append(ctr)

    return {"success_rate": float(np.mean(successes)), "mean_final_distance": float(np.mean(fin_dist)),
            "mean_return": float(np.mean(returns)), "mean_steps": float(np.mean(steps))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "checkpoints" / "sac_reach" / "best_model.zip"))
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--stats", default=None,
                        help="VecNormalize statistics (default: vecnormalize.pkl next to the model, if it exists)")
    args = parser.parse_args()

    env = ReachEnv(render_mode="human" if args.render else None)
    model = SAC.load(args.model)
    stats = Path(args.stats) if args.stats else Path(args.model).parent / "vecnormalize.pkl"
    if stats.exists():
        model = NormalizedPolicy(model, stats)
        print(f"using observation statistics from {stats}")
    else:
        print("no vecnormalize.pkl found: using raw observations (models trained before VecNormalize)")
    for name, policy in [("random", RandomPolicy(env)), ("trained", model)]:
        r = evaluate(policy, env, args.episodes, render=args.render and name == "trained")
        print(f"{name:8s} success {r['success_rate'] * 100:5.1f}%   final distance {r['mean_final_distance'] * 100:5.1f} cm"
              f"   return {r['mean_return']:7.1f}   steps {r['mean_steps']:5.1f}")
    env.close()
