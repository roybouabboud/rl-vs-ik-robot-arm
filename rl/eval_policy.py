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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reach_env import ReachEnv  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


class RandomPolicy:
    """Same predict() interface as an SB3 model, but acts randomly. The baseline to beat."""

    def __init__(self, env):
        self.action_space = env.action_space

    def predict(self, obs, deterministic=True):
        return self.action_space.sample(), None


def evaluate(model, env: ReachEnv, n_episodes: int = 50, seed: int = 10_000,
             render: bool = False) -> dict:
    """Run n_episodes, resetting episode i with seed + i, using the policy's deterministic action.

    Returns {"success_rate", "mean_final_distance", "mean_return", "mean_steps"}: success_rate in [0, 1],
    final distance in metres (from the last step's info), mean_steps = steps per episode.
    If render: call env.render() each step and sleep so it plays in real time."""
    # TODO 5
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "checkpoints" / "sac_reach" / "best_model.zip"))
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    env = ReachEnv(render_mode="human" if args.render else None)
    model = SAC.load(args.model)
    for name, policy in [("random", RandomPolicy(env)), ("trained", model)]:
        r = evaluate(policy, env, args.episodes, render=args.render and name == "trained")
        print(f"{name:8s} success {r['success_rate'] * 100:5.1f}%   final distance {r['mean_final_distance'] * 100:5.1f} cm"
              f"   return {r['mean_return']:7.1f}   steps {r['mean_steps']:5.1f}")
    env.close()
