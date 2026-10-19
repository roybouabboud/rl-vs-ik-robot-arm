"""Train a SAC policy on ReachEnv with Stable-Baselines3, logging to TensorBoard.

Step 6 of the roadmap (phase 2). Read rl/06_train_sac.md first,
then fill in the TODOs. Check your work with: python rl/check_train_sac.py

Train:   python rl/train_sac.py --steps 100000
Watch:   tensorboard --logdir runs     (then open http://localhost:6006)
"""

import argparse
import sys
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reach_env import ReachEnv  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"          # TensorBoard logs (gitignored)
CKPT_DIR = ROOT / "checkpoints"   # saved models (gitignored)


def make_envs(n_envs: int, seed: int):
    """Return (train_env, eval_env): two separate vectorized ReachEnvs.

    The training env has n_envs copies seeded from `seed`; the eval env has 1 copy seeded from
    seed + 1000, so evaluation episodes are ones the policy never trained on."""
    train_env = make_vec_env(ReachEnv, n_envs, seed=seed)
    eval_env = make_vec_env(ReachEnv, 1, seed=(seed+1000))
    return train_env, eval_env


def make_model(train_env, seed: int) -> SAC:
    """SAC with an MLP policy on train_env, seeded, logging to TensorBoard under RUNS_DIR."""
    return SAC("MlpPolicy", train_env, seed=seed, tensorboard_log=RUNS_DIR)


def make_eval_callback(eval_env, run_name: str, eval_freq: int = 5000) -> EvalCallback:
    """Every eval_freq steps: run 20 deterministic episodes on eval_env, log the results, and save the
    best model so far to CKPT_DIR / run_name (as best_model.zip)."""
    return EvalCallback(eval_env, eval_freq=eval_freq, n_eval_episodes=20, deterministic=True, log_path=CKPT_DIR/run_name, best_model_save_path=CKPT_DIR/run_name)


def train(steps: int, n_envs: int = 1, seed: int = 0, run_name: str = "sac_reach",
          eval_freq: int = 5000) -> SAC:
    """Build everything, train for `steps` environment steps, save the final model to
    CKPT_DIR / run_name / "final.zip", and return the model. Evaluate every eval_freq *env* steps
    (careful: EvalCallback counts vectorized steps, which are n_envs env steps each)."""
    train_env, eval_env = make_envs(n_envs, seed=seed)
    sac_model = make_model(train_env, seed=seed)
    callback = make_eval_callback(eval_env, run_name, eval_freq=max(eval_freq//n_envs, 1))
    sac_model.learn(steps, callback=callback, tb_log_name=run_name)
    sac_model.save(CKPT_DIR/run_name/"final")
    return sac_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default="sac_reach")
    args = parser.parse_args()
    train(args.steps, args.n_envs, args.seed, args.name)
    print(f"done. models in {CKPT_DIR / args.name}, logs in {RUNS_DIR}")
