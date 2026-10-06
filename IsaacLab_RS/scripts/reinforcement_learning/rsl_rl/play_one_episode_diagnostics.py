"""Evaluate the first episode of every environment for an RSL-RL checkpoint."""

import argparse
import csv
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate one episode per environment.")
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--seed", type=int, default=24)
parser.add_argument("--num_envs", type=int, default=100)
parser.add_argument("--output_csv", type=str, default=None)
parser.add_argument("--original_ant_reward", action="store_true", help="Score using the unmodified Ant reward terms.")
parser.add_argument("--track_course_progress", action="store_true", help="Record target visits without terminating the episode.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

simulation_app = AppLauncher(args_cli).app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    if args_cli.original_ant_reward:
        from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import RewardsCfg
        env_cfg.rewards = RewardsCfg()
        print("[INFO] Scoring with the original Isaac-Ant-v0 reward terms.", flush=True)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    agent_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    checkpoint = retrieve_file_path(args_cli.checkpoint)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    rewards = torch.zeros(env.num_envs, device=env.device)
    steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    finished = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    timed_out = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    manager = env.unwrapped.termination_manager
    course_eval = "course_goal" in manager.active_terms
    end_reasons = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    term_names = manager.active_terms
    if args_cli.track_course_progress:
        from isaaclab_tasks.manager_based.classic.ant_terrain.walk_mdp import course_goal_reached
        if not {"ground_probe", "foot_contacts"}.issubset(env.unwrapped.scene.sensors):
            raise ValueError("Progress tracking requires ground_probe and foot_contacts sensors.")
        reached_goal = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        first_goal_steps = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        max_forward = torch.zeros(env.num_envs, device=env.device)
    obs = env.get_observations()

    with torch.inference_mode():
        while not bool(finished.all()):
            if args_cli.track_course_progress:
                active = ~finished
                at_goal = course_goal_reached(env.unwrapped, distance=3.3, half_length=3.7)
                first_visit = active & at_goal & ~reached_goal
                first_goal_steps[first_visit] = steps[first_visit]
                reached_goal |= active & at_goal
                local_x = env.unwrapped.scene["robot"].data.root_pos_w[:, 0] - env.unwrapped.scene.env_origins[:, 0]
                max_forward[active] = torch.maximum(max_forward[active], local_x[active])
            actions = policy(obs)
            obs, step_rewards, dones, extras = env.step(actions)
            active = ~finished
            rewards[active] += step_rewards[active]
            steps[active] += 1
            newly_finished = active & dones.bool()
            for term_id, term_name in enumerate(term_names):
                ended = newly_finished & manager.get_term(term_name)
                end_reasons[ended] = term_id
            if "time_outs" in extras:
                timed_out[newly_finished] = extras["time_outs"][newly_finished].bool()
            finished |= dones.bool()

    print(f"[INFO] All {env.num_envs} environments finished their first episode.")
    print(f"[INFO] Completed first episodes: {int(finished.sum())}/{env.num_envs}")
    print(f"[RESULT] Episode reward total: mean={rewards.mean().item():.6f}, std={rewards.std(unbiased=False).item():.6f}")
    print(f"[RESULT] Episode steps: mean={steps.float().mean().item():.6f}, std={steps.float().std(unbiased=False).item():.6f}")
    print(f"[RESULT] Timeout fraction: {timed_out.float().mean().item():.6f}")
    for term_id, name in enumerate(term_names):
        print(f"[RESULT] End reason {name}: {int((end_reasons == term_id).sum())}/{env.num_envs}")
    if args_cli.track_course_progress:
        count = int(reached_goal.sum())
        print(f"[RESULT] Visited course goal during episode: {count}/{env.num_envs}")
        print(f"[RESULT] Maximum forward distance (pre-step): mean={max_forward.mean().item():.3f} m")
        if count:
            after_goal_s = (steps[reached_goal] - first_goal_steps[reached_goal]).float() * env.unwrapped.step_dt
            print(f"[RESULT] Time remaining after first goal visit: mean={after_goal_s.mean().item():.3f} s")
            print(f"[RESULT] Timeout after goal visit: {int((timed_out & reached_goal).sum())}/{count}")
    if course_eval:
        successes = end_reasons == term_names.index("course_goal")
        exits = end_reasons == term_names.index("course_out_of_bounds")
        print(f"[RESULT] Course success: {int(successes.sum())}/{env.num_envs} ({successes.float().mean().item():.2%})")
        print(f"[RESULT] Course exit: {int(exits.sum())}/{env.num_envs}")
        terrain_cfg = env.unwrapped.scene.terrain.cfg.terrain_generator
        shape_names = list(terrain_cfg.sub_terrains)
        proportions = torch.tensor([c.proportion for c in terrain_cfg.sub_terrains.values()], device=env.device)
        cumulative = (proportions / proportions.sum()).cumsum(0)
        columns = env.unwrapped.scene.terrain.terrain_types
        shape_ids = torch.searchsorted(cumulative, columns.float() / terrain_cfg.num_cols + 0.001)
        for shape_id, name in enumerate(shape_names):
            mask = shape_ids == shape_id
            if bool(mask.any()):
                print(f"[RESULT] {name}: success={int((successes & mask).sum())}/{int(mask.sum())}, exit={int((exits & mask).sum())}, timeout={int((timed_out & mask).sum())}")
    if args_cli.output_csv is not None:
        output_path = Path(args_cli.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        terrain = env.unwrapped.scene.terrain
        terrain_types = terrain.terrain_types.cpu().tolist()
        terrain_levels = terrain.terrain_levels.cpu().tolist()
        with output_path.open("w", newline="") as file:
            writer = csv.writer(file)
            header = ["env_id", "terrain_type", "terrain_level", "reward", "steps", "timed_out", "end_reason", "terrain_shape"]
            if args_cli.track_course_progress:
                header += ["visited_course_goal", "first_goal_step", "max_forward_m"]
            writer.writerow(header)
            for env_id in range(env.num_envs):
                row = [
                    env_id,
                    terrain_types[env_id],
                    terrain_levels[env_id],
                    rewards[env_id].item(),
                    steps[env_id].item(),
                    int(timed_out[env_id].item()),
                    term_names[end_reasons[env_id].item()] if end_reasons[env_id] >= 0 else "unknown",
                    shape_names[shape_ids[env_id].item()] if course_eval else "unknown",
                ]
                if args_cli.track_course_progress:
                    row += [int(reached_goal[env_id].item()), first_goal_steps[env_id].item(), max_forward[env_id].item()]
                writer.writerow(row)
        print(f"[INFO] Per-environment results saved to: {output_path}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
