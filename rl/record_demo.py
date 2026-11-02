"""Record a GIF of a trained policy reaching a chain of random targets, for the README.

Run from the repo root:
    python rl/record_demo.py --model checkpoints/<run>/final.zip --out media/reach_demo.gif
Uses vecnormalize.pkl next to the model if present. After each target, a new one appears without
resetting the arm, so the clip shows one continuous motion. With --best-of N, each next target is the
one (of N random candidates at least 25 cm away) that the policy reaches and holds fastest from where
the arm is, so the clip is a highlight reel rather than a typical sample.
"""

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image
from stable_baselines3 import SAC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_policy import NormalizedPolicy  # noqa: E402
from reach_env import ReachEnv  # noqa: E402


def snapshot(env: ReachEnv):
    """Everything needed to rewind the env to this moment."""
    data = mujoco.MjData(env.model)
    mujoco.mj_copyData(data, env.model, env.data)
    return data, env.prev_action.copy()


def restore(env: ReachEnv, snap):
    mujoco.mj_copyData(env.data, env.model, snap[0])
    env.prev_action = snap[1].copy()


def reach(policy, env: ReachEnv, target: np.ndarray, frame=None, every: int = 1) -> tuple[bool, int]:
    """Give the env `target` mid-episode (the arm stays put) and play until held or out of steps.
    Returns (held, steps). If `frame` is given, it is called every `every`-th step to record."""
    env.target = target.copy()
    env.data.mocap_pos[0] = target
    env.in_tol_steps = 0
    env.steps = 0
    mujoco.mj_forward(env.model, env.data)
    obs = env._get_obs()
    for t in range(env.max_steps):
        action, _ = policy.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        if frame is not None and t % every == 0:
            frame()
        if terminated or truncated:
            break
    return bool(info["is_success"]), t + 1


def pick_target(policy, env: ReachEnv, best_of: int, min_dist: float = 0.25) -> np.ndarray:
    """Of `best_of` random targets, the one held fastest from the current state (env is left unchanged).
    With best_of=1, just a random target."""
    if best_of <= 1:
        return env._sample_target()
    snap, best = snapshot(env), None
    for _ in range(best_of):
        target = env._sample_target()
        if np.linalg.norm(target - env.data.site_xpos[env.ee_id]) < min_dist:
            continue  # skip targets the hand is already next to
        held, steps = reach(policy, env, target)
        restore(env, snap)
        if held and (best is None or steps < best[1]):
            best = (target, steps)
    return best[0] if best is not None else env._sample_target()


def record(policy, env: ReachEnv, n_targets: int, seed: int, every: int, size: tuple[int, int],
           pause: int, best_of: int = 1) -> tuple[list[Image.Image], int]:
    """Play n_targets targets in a row; return (RGB frames, number held). Keeps every `every`-th env step,
    and repeats the last frame `pause` times after each target so the viewer can see it."""
    renderer = mujoco.Renderer(env.model, height=size[1], width=size[0])
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.0, 0.3]
    cam.distance, cam.azimuth, cam.elevation = 1.75, 140.0, -25.0
    frames = []

    def frame():
        renderer.update_scene(env.data, camera=cam)
        frames.append(Image.fromarray(renderer.render()))

    env.reset(seed=seed)
    held = 0
    for _ in range(n_targets):
        target = pick_target(policy, env, best_of)
        held += reach(policy, env, target, frame, every)[0]
        frames += [frames[-1]] * pause
    renderer.close()
    return frames, held


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", default="media/reach_demo.gif")
    parser.add_argument("--targets", type=int, default=10)
    parser.add_argument("--best-of", type=int, default=12, help="candidate targets per step; 1 = purely random")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--every", type=int, default=3, help="keep every n-th env step (one step = 0.02 s)")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--colors", type=int, default=96)
    args = parser.parse_args()

    env = ReachEnv()
    policy = SAC.load(args.model)
    stats = Path(args.model).parent / "vecnormalize.pkl"
    if stats.exists():
        policy = NormalizedPolicy(policy, stats)
    frames, held = record(policy, env, args.targets, args.seed, args.every, (args.width, args.height),
                          pause=10, best_of=args.best_of)

    # one shared palette for the whole clip, built from frames spread across it so that small, rare colours
    # (the red target) get palette entries too
    picks = frames[::max(1, len(frames) // 12)]
    strip = Image.new("RGB", (args.width * len(picks), args.height))
    for i, f in enumerate(picks):
        strip.paste(f, (i * args.width, 0))
    palette = strip.quantize(colors=args.colors, method=Image.Quantize.FASTOCTREE)
    frames = [f.quantize(palette=palette) for f in frames]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ms = round(1000 * args.every * env.frame_skip * env.model.opt.timestep)  # real time
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=ms, loop=0, optimize=True)
    print(f"{out}: {len(frames)} frames, {held}/{args.targets} targets held, {out.stat().st_size / 1e6:.1f} MB")
    env.close()
