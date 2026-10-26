# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay recorded demos headless and report whether each one succeeds again.

Same replay as ``scripts/tools/replay_demos.py`` (reset to the stored initial
state, step the stored actions), but after the last action the task's own
success term is evaluated and the truck-box distance is printed, so the check
from the Week 08 slides ("does the saved demo succeed again?") can run without
a screen. Nothing is written to the dataset.

Usage (from IsaacLab_RS):
    ./isaaclab.sh -p coursework/toy2box-bc/replay_check.py --dataset_file datasets/team03_kim_s1.hdf5 --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay demos and check success.")
parser.add_argument("--task", type=str, default=None, help="Task name (defaults to the one stored in the file).")
parser.add_argument("--dataset_file", type=str, required=True, help="HDF5 file with recorded demos.")
parser.add_argument("--select_episodes", type=int, nargs="+", default=[], help="Episode indices to replay.")
parser.add_argument("--settle_steps", type=int, default=10, help="Extra zero-action steps before judging success.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from isaaclab.utils.datasets import HDF5DatasetFileHandler

from isaaclab_tasks.utils import parse_env_cfg


def main():
    handler = HDF5DatasetFileHandler()
    handler.open(args_cli.dataset_file)
    env_name = args_cli.task or handler.get_env_name()
    episode_names = list(handler.get_episode_names())
    if args_cli.select_episodes:
        episode_names = [episode_names[i] for i in args_cli.select_episodes]

    env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=1)
    env_cfg.recorders = {}
    success_term = env_cfg.terminations.success
    env_cfg.terminations = {}
    env = gym.make(env_name, cfg=env_cfg).unwrapped

    results = []
    for name in episode_names:
        ep = handler.load_episode(name, env.device)
        env.reset_to(ep.get_initial_state(), torch.tensor([0], device=env.device), is_relative=True)
        n = 0
        action = ep.get_next_action()
        while action is not None:
            env.step(action.unsqueeze(0))
            n += 1
            action = ep.get_next_action()
        zero = torch.zeros((1, env.action_space.shape[1]), device=env.device)
        # keep the last gripper command while the scene settles
        zero[0, -1] = ep.data["actions"][-1, -1]
        for _ in range(args_cli.settle_steps):
            env.step(zero)
        ok = bool(success_term.func(env, **success_term.params)[0])
        truck = env.scene["toy_truck"].data.root_pos_w[0]
        box = env.scene["box"].data.root_pos_w[0]
        d_xy = float(torch.linalg.norm((truck - box)[:2]))
        dz = float(truck[2] - box[2])
        results.append((name, n, ok, d_xy, dz))
        print(f"{name:<8s} steps={n:4d} success={ok}  truck-box xy={d_xy:.3f} m  dz={dz:+.3f} m")

    n_ok = sum(r[2] for r in results)
    print(f"\nReplay success: {n_ok} / {len(results)}")
    for r in results:
        if not r[2]:
            print(f"  failed: {r[0]}")
    handler.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
