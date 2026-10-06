# robot_rl_vla

A simulated robot arm (UR5e in MuJoCo) that moves to any point I prompt it to.
Simulation and software only.

## Roadmap

**Phase 1: kinematics and control with MuJoCo** (MuJoCo does the math; I learn the API)
- [x] 1. Forward kinematics: `mj_kinematics`, site poses, workspace sampling (`kinematics/`)
- [x] 2. Jacobians with `mj_jacSite`
- [x] 3. Inverse kinematics on top of MuJoCo's Jacobian; type `x y z`, the arm goes there
- [x] 3b. Full-pose IK (position + orientation): 6×6 Jacobian, orientation error via `mju_subQuat`
- [ ] 4. Trajectories and control: smooth motion, torque control with gravity compensation (`qfrc_bias`)

**Phase 2: reinforcement learning**
- [ ] Gymnasium "reach any point" env, PPO/SAC, benchmark against the IK solver

**Phase 3: VLM / VLA**
- [ ] Language ("go to the red cube") to 3D target via a VLM, then IK
- [ ] End-to-end VLA policy

## Setup

```bash
pip install -r requirements.txt
git clone https://github.com/google-deepmind/mujoco_menagerie.git   # robot models, not tracked here
```

Run scripts from the repo root, e.g. `python kinematics/check_fk.py`.

## Layout

- `kinematics/`: phase 1 code and guides (`01_forward_kinematics.md`, ...)
- `test/`: early MuJoCo experiments
- `envs/`, `rl/`, `models/`, `experiments/`, `scripts/`: reserved for later phases
