# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a robomimic policy and classify every failed rollout by stage.

Same rollout loop, seeding and environment setup as
``scripts/imitation_learning/robomimic/play.py`` so the success rate is the
one ``play.py`` would report. In addition, every step records the end-effector,
toy truck, box and gripper state so a failure can be assigned to one stage:

    approach   the hand never came close to the truck
    grasp      the hand reached the truck but never held it
    transport  the truck was held, then lost before it was over the box
    place      the truck got to the box but the success condition never held
               (gripper not opened, truck on the rim, wrong height)

Results are written as JSON next to the log so they can be aggregated later.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a robomimic policy and diagnose failures.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Pytorch model checkpoint to load.")
parser.add_argument("--horizon", type=int, default=800, help="Step horizon of each rollout.")
parser.add_argument("--num_rollouts", type=int, default=1, help="Number of rollouts.")
parser.add_argument("--seed", type=int, default=101, help="Random seed.")
parser.add_argument("--output", type=str, default=None, help="JSON file for per-trial results.")
parser.add_argument("--save_traj", type=str, default=None, help="Optional .npz with per-step trajectories.")
parser.add_argument(
    "--norm_factor_min", type=float, default=None, help="Optional: minimum value of the normalization factor."
)
parser.add_argument(
    "--norm_factor_max", type=float, default=None, help="Optional: maximum value of the normalization factor."
)
parser.add_argument("--enable_pinocchio", default=False, action="store_true", help="Enable Pinocchio.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import copy
import gymnasium as gym
import json
import numpy as np
import os
import random
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401
    import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401

from isaaclab_tasks.utils import parse_env_cfg

# Stage thresholds (metres). xy_threshold matches the success term of the task.
NEAR_TRUCK = 0.08  # hand counted as "reached the truck" below this hand-truck distance
NEAR_BOX_XY = 0.12  # truck counted as "at the box" below this truck-box xy distance
GRIPPER_OPEN = 0.6  # gripper_pos above this is open (open 0.994 / close 0.20)


def _vec(obs, key):
    return torch.squeeze(obs[key]).detach().cpu().numpy().astype(float)


def classify(rec: dict) -> dict:
    """Assign a stage to one rollout from its per-step record."""
    eef = np.asarray(rec["eef_pos"])
    truck = np.asarray(rec["truck_pos"])
    box = np.asarray(rec["box_pos"])
    grip = np.asarray(rec["gripper_pos"])
    grasp = np.asarray(rec["grasp"], dtype=bool)

    d_et = np.linalg.norm(eef - truck, axis=1)
    d_tb_xy = np.linalg.norm((truck - box)[:, :2], axis=1)
    gripper_open = grip.mean(axis=1) > GRIPPER_OPEN

    ever_grasped = bool(grasp.any())
    first_grasp = int(np.argmax(grasp)) if ever_grasped else None
    grasp_episodes = int(np.sum(np.diff(grasp.astype(int)) == 1))

    out = {
        "min_hand_truck_dist": float(d_et.min()),
        "ever_grasped": ever_grasped,
        "first_grasp_step": first_grasp,
        "grasp_attempts": grasp_episodes,
        "max_truck_z": float(truck[:, 2].max()),
        "final_truck_box_xy": float(d_tb_xy[-1]),
        "final_truck_box_dz": float(truck[-1, 2] - box[-1, 2]),
        "final_gripper_open": bool(gripper_open[-1]),
        "truck_dropped": bool(rec["truck_dropped"]),
    }

    if rec["success"]:
        out["stage"] = "success"
        return out
    if not ever_grasped:
        out["stage"] = "approach" if d_et.min() > NEAR_TRUCK else "grasp"
        return out
    # held at some point: did the truck get to the box while/after being held?
    reached_box = bool((d_tb_xy[first_grasp:] < NEAR_BOX_XY).any())
    if rec["truck_dropped"] or not reached_box:
        out["stage"] = "transport"
    else:
        out["stage"] = "place"
    return out


def rollout(policy, env, success_term, horizon, device):
    policy.start_episode()
    obs_dict, _ = env.reset()
    rec = dict(eef_pos=[], truck_pos=[], box_pos=[], gripper_pos=[], grasp=[], actions=[])
    truck_dropped = False
    success = False
    steps = 0

    for _ in range(horizon):
        obs = copy.deepcopy(obs_dict["policy"])
        for ob in obs:
            obs[ob] = torch.squeeze(obs[ob])

        if hasattr(env.cfg, "image_obs_list"):
            for image_name in env.cfg.image_obs_list:
                if image_name in obs_dict["policy"].keys():
                    image = torch.squeeze(obs_dict["policy"][image_name])
                    image = image.permute(2, 0, 1).clone().float()
                    image = image / 255.0
                    image = image.clip(0.0, 1.0)
                    obs[image_name] = image

        rec["eef_pos"].append(_vec(obs, "eef_pos").tolist())
        rec["truck_pos"].append(_vec(obs, "toy_truck_positions").tolist())
        rec["box_pos"].append(_vec(obs, "box_positions").tolist())
        rec["gripper_pos"].append(_vec(obs, "gripper_pos").tolist())
        grasp_sig = obs_dict.get("subtask_terms", {}).get("grasp")
        rec["grasp"].append(bool(torch.squeeze(grasp_sig).item()) if grasp_sig is not None else False)

        actions = policy(obs)
        if args_cli.norm_factor_min is not None and args_cli.norm_factor_max is not None:
            actions = (
                (actions + 1) * (args_cli.norm_factor_max - args_cli.norm_factor_min)
            ) / 2 + args_cli.norm_factor_min
        actions = torch.from_numpy(actions).to(device=device).view(1, env.action_space.shape[1])
        rec["actions"].append(actions.squeeze(0).tolist())

        obs_dict, _, terminated, truncated, _ = env.step(actions)
        steps += 1

        if bool(success_term.func(env, **success_term.params)[0]):
            success = True
            break
        elif terminated or truncated:
            truck_dropped = bool(terminated)
            break

    rec["success"] = success
    rec["truck_dropped"] = truck_dropped
    rec["steps"] = steps
    return rec


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.terminations.time_out = None
    env_cfg.recorders = None
    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)
    env.seed(args_cli.seed)

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    results = []
    trials = []
    trajs = {}
    for trial in range(args_cli.num_rollouts):
        print(f"[INFO] Starting trial {trial}")
        policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=args_cli.checkpoint, device=device)
        rec = rollout(policy, env, success_term, args_cli.horizon, device)
        results.append(rec["success"])
        info = classify(rec)
        info.update(trial=trial, success=rec["success"], steps=rec["steps"])
        trials.append(info)
        print(f"[INFO] Trial {trial}: {rec['success']}  stage={info['stage']} steps={rec['steps']}\n")
        if args_cli.save_traj:
            for k in ("eef_pos", "truck_pos", "box_pos", "gripper_pos", "grasp", "actions"):
                trajs[f"trial{trial}/{k}"] = np.asarray(rec[k])

    # Same three lines as play.py, so existing log parsers keep working.
    print(f"\nSuccessful trials: {results.count(True)}, out of {len(results)} trials")
    print(f"Success rate: {results.count(True) / len(results)}")
    print(f"Trial Results: {results}\n")

    stages = {}
    for t in trials:
        stages[t["stage"]] = stages.get(t["stage"], 0) + 1
    print("Stage counts:", json.dumps(stages))

    summary = {
        "task": args_cli.task,
        "checkpoint": os.path.abspath(args_cli.checkpoint),
        "seed": args_cli.seed,
        "num_rollouts": args_cli.num_rollouts,
        "horizon": args_cli.horizon,
        "successes": results.count(True),
        "success_rate": results.count(True) / len(results),
        "stage_counts": stages,
        "thresholds": {"near_truck": NEAR_TRUCK, "near_box_xy": NEAR_BOX_XY, "gripper_open": GRIPPER_OPEN},
        "trials": trials,
    }
    if args_cli.output:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.output)), exist_ok=True)
        with open(args_cli.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[INFO] Wrote {args_cli.output}")
    if args_cli.save_traj:
        np.savez_compressed(args_cli.save_traj, **trajs)
        print(f"[INFO] Wrote {args_cli.save_traj}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
