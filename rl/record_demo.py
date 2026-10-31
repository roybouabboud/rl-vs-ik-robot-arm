"""Record a GIF of a trained policy reaching a chain of random targets, for the README.

Run from the repo root:
    python rl/record_demo.py --model checkpoints/<run>/final.zip --out media/demo.gif
Uses vecnormalize.pkl next to the model if present. After each target is held, a new one appears
without resetting the arm, so the clip shows one continuous motion.
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


def new_target(env: ReachEnv):
    """Give the env a fresh target mid-episode (same sampling as reset, but the arm stays put)."""
    env.target = env._sample_target()
    env.data.mocap_pos[0] = env.target
    env.in_tol_steps = 0
    env.steps = 0
    mujoco.mj_forward(env.model, env.data)
    return env._get_obs()


def record(policy, env: ReachEnv, n_targets: int, seed: int, every: int, size: tuple[int, int],
           pause: int, colors: int = 64) -> tuple[list[Image.Image], int]:
    """Play n_targets targets in a row; return (frames, number reached). Keeps every `every`-th env step,
    and holds the arm still for `pause` frames after each success so the viewer can see it."""
    renderer = mujoco.Renderer(env.model, height=size[1], width=size[0])
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.0, 0.3]
    cam.distance, cam.azimuth, cam.elevation = 1.75, 140.0, -25.0

    def frame():
        renderer.update_scene(env.data, camera=cam)
        return Image.fromarray(renderer.render()).quantize(colors=colors, method=Image.Quantize.FASTOCTREE)  # small palette keeps the GIF light;
        # octree keeps small distinct colours such as the red target

    frames, reached = [], 0
    obs, _ = env.reset(seed=seed)
    for _ in range(n_targets):
        for t in range(env.max_steps):
            action, _ = policy.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if t % every == 0:
                frames.append(frame())
            if terminated or truncated:
                break
        reached += bool(info["is_success"])
        frames += [frame()] * pause
        obs = new_target(env)
    renderer.close()
    return frames, reached


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", default="media/demo.gif")
    parser.add_argument("--targets", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--every", type=int, default=3, help="keep every n-th env step (one step = 0.02 s)")
    parser.add_argument("--width", type=int, default=400)
    parser.add_argument("--height", type=int, default=300)
    parser.add_argument("--colors", type=int, default=64)
    args = parser.parse_args()

    env = ReachEnv()
    policy = SAC.load(args.model)
    stats = Path(args.model).parent / "vecnormalize.pkl"
    if stats.exists():
        policy = NormalizedPolicy(policy, stats)
    frames, reached = record(policy, env, args.targets, args.seed, args.every, (args.width, args.height),
                             pause=6, colors=args.colors)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ms = round(1000 * args.every * env.frame_skip * env.model.opt.timestep)  # real time
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=ms, loop=0, optimize=True)
    print(f"{out}: {len(frames)} frames, {reached}/{args.targets} targets reached, "
          f"{out.stat().st_size / 1e6:.1f} MB")
    env.close()
