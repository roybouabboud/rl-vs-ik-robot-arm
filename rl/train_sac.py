"""Train a SAC policy on ReachEnv with Stable-Baselines3, logging to TensorBoard.

Train:   python rl/train_sac.py --steps 100000
Watch:   tensorboard --logdir runs     (then open http://localhost:6006)
"""

import argparse
import sys
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from collections import deque
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reach_env import ReachEnv  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"          # TensorBoard logs (gitignored)
CKPT_DIR = ROOT / "checkpoints"   # saved models (gitignored)

class CurriculumCallback(BaseCallback):
    """Curriculum on the training envs: start at levels[0] = (success_tol, overshoot_weight), and move to
    the next level once the success rate over the last `window` training episodes reaches `threshold`.
    Levels are consumed (popped) as training advances; the eval env keeps its own defaults."""
    def __init__(self, levels, threshold=0.8, window=100):
        super().__init__()
        self.levels = levels
        self.threshold = threshold
        self.window = window
        self.deque = deque(maxlen=window)
        self.suc_tol = 0.0
        self.overshoot = 0.0

    def _on_training_start(self):
        self.suc_tol, self.overshoot = self.levels.pop(0)
        self.training_env.env_method("set_wrapper_attr", "success_tol", self.suc_tol)
        self.training_env.env_method("set_wrapper_attr", "overshoot_weight", self.overshoot)

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.deque.append(info["is_success"])
        if len(self.deque) == self.deque.maxlen and np.mean(self.deque) >= self.threshold and self.levels:
            self.suc_tol, self.overshoot = self.levels.pop(0)
            self.training_env.env_method("set_wrapper_attr", "success_tol", self.suc_tol)
            self.training_env.env_method("set_wrapper_attr", "overshoot_weight", self.overshoot)
            self.deque.clear()
        self.logger.record("curriculum/success_tol", self.suc_tol)
        self.logger.record("curriculum/overshoot_weight", self.overshoot)
        return True
            

def make_envs(n_envs: int, seed: int):
    """Return (train_env, eval_env): two separate vectorized ReachEnvs.

    The training env has n_envs copies seeded from `seed`; the eval env has 1 copy seeded from
    seed + 1000, so evaluation episodes are ones the policy never trained on."""
    train_env = make_vec_env(ReachEnv, n_envs, seed=seed)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False)
    eval_env = make_vec_env(ReachEnv, 1, seed=(seed+1000))
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
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
    (careful: EvalCallback counts vectorized steps, which are n_envs env steps each). Training uses a
    success_tol / overshoot_weight curriculum (CurriculumCallback)."""
    train_env, eval_env = make_envs(n_envs, seed=seed)
    sac_model = make_model(train_env, seed=seed)
    curriculum_callback = CurriculumCallback([(0.05, 0.0), (0.04, 0.0), (0.03, 0.0), (0.02, 0.0)])
    eval_callback = make_eval_callback(eval_env, run_name, eval_freq=max(eval_freq//n_envs, 1))
    sac_model.learn(steps, callback=list([eval_callback, curriculum_callback]), tb_log_name=run_name)
    train_env.save(CKPT_DIR/run_name/"vecnormalize.pkl")
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
