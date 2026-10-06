# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument(
    "--camera_mode", choices=("follow", "overview"), default="follow",
    help="Follow the first Ant, or show all environments from a fixed overview camera.",
)
parser.add_argument(
    "--log_resets", action="store_true",
    help="Print the termination conditions for each automatic reset (manager-based environments).",
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


def overview_camera_pose(origins, terrain_size, continuous_course=False):
    """Frame the active terrain tiles, including the Ant course ahead of spawn."""
    lower = [min(origin[axis] for origin in origins) for axis in range(3)]
    upper = [max(origin[axis] for origin in origins) for axis in range(3)]
    if continuous_course:
        # The Ant spawns at the first 8 m tile's center, also in longer courses.
        lower[0] -= 4.0
        upper[0] += terrain_size[0] - 4.0
    else:
        lower[0] -= terrain_size[0] / 2
        upper[0] += terrain_size[0] / 2
    lower[1] -= terrain_size[1] / 2
    upper[1] += terrain_size[1] / 2
    lookat = ((lower[0] + upper[0]) / 2, (lower[1] + upper[1]) / 2, upper[2] + 0.5)
    distance = max(10.0, 1.5 * max(upper[0] - lower[0], upper[1] - lower[1]))
    eye = (lookat[0] - 0.25 * distance, lookat[1] - 0.25 * distance, lookat[2] + distance)
    return eye, lookat


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # Set the Ant camera after Hydra parsing: asset_name defaults to None,
    # so overriding it with a string through Hydra fails config type checks.
    if args_cli.camera_mode == "overview":
        env_cfg.viewer.origin_type = "world"
    elif task_name.startswith("Isaac-Ant-"):
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = (-4.0, 4.0, 2.5)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.5)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.camera_mode == "overview":
        base_env = env.unwrapped
        camera = base_env.viewport_camera_controller
        if camera is not None:
            terrain = base_env.scene.terrain
            generator = terrain.cfg.terrain_generator if terrain is not None else None
            terrain_size = generator.size if generator is not None else (env_cfg.scene.env_spacing,) * 2
            eye, lookat = overview_camera_pose(
                base_env.scene.env_origins.detach().cpu().tolist(), terrain_size,
                continuous_course=task_name.startswith(("Isaac-Ant-Continuous", "Isaac-Ant-Common")),
            )
            camera.update_view_location(eye=eye, lookat=lookat)
            print(f"[INFO] Overview camera: eye={eye}, lookat={lookat}")

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    if args_cli.log_resets:
        termination_manager = getattr(env.unwrapped, "termination_manager", None)
        if termination_manager is None:
            raise ValueError("--log_resets requires a manager-based environment.")
        episode_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.unwrapped.device)
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            if args_cli.log_resets:
                episode_steps += 1
                # Manager term buffers retain the terminal flags after auto-reset;
                # robot/sensor data already describe the reset state at this point.
                for env_id in dones.nonzero(as_tuple=True)[0].tolist():
                    reasons = [
                        name for name in termination_manager.active_terms
                        if bool(termination_manager.get_term(name)[env_id])
                    ]
                    seconds = episode_steps[env_id].item() * dt
                    detail = ""
                    if "flight" in reasons:
                        flight_term = termination_manager.get_term_cfg("flight").func
                        if hasattr(flight_term, "last_has_landed"):
                            landed = int(flight_term.last_has_landed[env_id].item())
                            air_s = flight_term.last_air_time_s[env_id].item()
                            detail = f" first_contact={landed} all_feet_air_s={air_s:.2f}"
                    if "long_course_progress" in termination_manager.active_terms:
                        tracker = termination_manager.get_term_cfg("long_course_progress").func
                        frontier = tracker.last_grounded_max_x[env_id].item()
                        detail += f" grounded_frontier_m={frontier:.2f} tiles_passed={int((frontier + 4.0) // 8.0)}"
                    print(f"[RESET] env={env_id} episode_s={seconds:.2f} reasons={','.join(reasons)}{detail}", flush=True)
                episode_steps[dones.bool()] = 0
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
