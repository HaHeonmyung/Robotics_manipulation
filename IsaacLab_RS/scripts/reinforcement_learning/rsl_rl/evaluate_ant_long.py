"""Evaluate sustained Ant locomotion without confusing survival with progress."""

import argparse
import csv
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate first episodes on the long Recovery course.")
parser.add_argument("--task", default="Isaac-Ant-Continuous-Wide-Ray-Recovery-Long-Eval-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--seed", type=int, default=24)
parser.add_argument("--num_envs", type=int, default=100)
parser.add_argument("--output_csv", type=str, default=None)
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
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = agent_cfg.device = args_cli.device
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    base = env.unwrapped
    manager = base.termination_manager
    tracker = manager.get_term_cfg("long_course_progress").func
    names = [name for name in manager.active_terms if name != "long_course_progress"]
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=base.device)
    runner.load(retrieve_file_path(args_cli.checkpoint))
    policy = runner.get_inference_policy(device=base.device)
    obs = env.get_observations()
    count, device = env.num_envs, base.device
    reward = torch.zeros(count, dtype=torch.float64, device=device)
    steps = torch.zeros(count, dtype=torch.long, device=device)
    finished = torch.zeros(count, dtype=torch.bool, device=device)
    final_x = torch.zeros(count, device=device)
    max_x = torch.zeros_like(final_x)
    grounded_max_x = torch.zeros_like(final_x)
    terminal_flags = {name: torch.zeros_like(finished) for name in names}
    horizon = base.max_episode_length
    # First-tile center is local x=0; edges to cross are 4, 12, 20, ... m.
    tile_size = next(iter(env_cfg.scene.terrain.terrain_generator.sub_terrains.values())).tile_size
    marks = sorted({s for s in (16.0, 32.0, 48.0, float(env_cfg.episode_length_s)) if s <= env_cfg.episode_length_s})
    snapshots = {round(s / base.step_dt): (s, torch.zeros_like(final_x)) for s in marks}
    late_window_step = max(0, horizon - math.ceil(2.0 / base.step_dt))
    late_window_x = torch.zeros_like(final_x)

    with torch.inference_mode():
        for tick in range(1, horizon + 1):
            active = ~finished
            obs, step_reward, dones, _ = env.step(policy(obs))
            reward[active] += step_reward[active]
            steps[active] += 1
            # These snapshots were taken BEFORE automatic reset in env.step().
            final_x[active] = tracker.last_x[active]
            max_x[active] = tracker.last_max_x[active]
            grounded_max_x[active] = tracker.last_grounded_max_x[active]
            newly_finished = active & dones.bool()
            for name in names:
                terminal_flags[name][newly_finished] = manager.get_term(name)[newly_finished]
            finished |= newly_finished
            if tick == late_window_step:
                late_window_x.copy_(grounded_max_x)
            if tick in snapshots:
                seconds, snapshot = snapshots[tick]
                snapshot.copy_(grounded_max_x)
                reached = steps >= tick
                print(
                    f"[PROGRESS] {seconds:.0f}s: reached={int(reached.sum())}/{count}, "
                    f"grounded frontier mean(all)={grounded_max_x.mean().item():.3f} m",
                    flush=True,
                )
            if bool(finished.all()):
                # Later snapshot columns represent the terminal frontier for
                # agents that failed before their requested checkpoint time.
                for future_step, (_, snapshot) in snapshots.items():
                    if future_step > tick:
                        snapshot.copy_(grounded_max_x)
                break

    duration = steps.double() * base.step_dt
    tiles_passed = ((grounded_max_x + tile_size / 2) / tile_size).floor().clamp_min(0).long()
    first_names = list(env_cfg.scene.terrain.terrain_generator.sub_terrains)
    proportions = torch.tensor([
        cfg.proportion for cfg in env_cfg.scene.terrain.terrain_generator.sub_terrains.values()
    ], device=device)
    cumulative = (proportions / proportions.sum()).cumsum(0)
    column_quantiles = (
        base.scene.terrain.terrain_types.float() / env_cfg.scene.terrain.terrain_generator.num_cols + 0.001
    )
    shape_ids = torch.searchsorted(cumulative, column_quantiles)
    reached_16 = steps >= round(16.0 / base.step_dt)
    frontier_16 = snapshots.get(round(16.0 / base.step_dt), (16.0, grounded_max_x))[1]
    post_16 = (grounded_max_x - frontier_16).clamp_min(0)
    timed_out = terminal_flags.get("time_out", torch.zeros_like(finished))
    failed = torch.zeros_like(finished)
    for name in names:
        if not manager.get_term_cfg(name).time_out:
            failed |= terminal_flags[name]
    full_time = timed_out & ~failed
    progressed_after_16 = reached_16 & (post_16 >= 1.0)
    moving_at_end = full_time & ((grounded_max_x - late_window_x) >= 0.10)
    print(f"[INFO] Completed first episodes: {int(finished.sum())}/{count}")
    print(f"[RESULT] Episode reward total: mean={reward.mean().item():.6f}, std={reward.std(unbiased=False).item():.6f}")
    print(f"[RESULT] Episode seconds: mean={duration.mean().item():.3f}, std={duration.std(unbiased=False).item():.3f}")
    print(f"[RESULT] Grounded maximum forward distance: mean={grounded_max_x.mean().item():.3f} m, std={grounded_max_x.std(unbiased=False).item():.3f} m")
    print(f"[RESULT] Tiles passed (grounded frontier): mean={tiles_passed.float().mean().item():.2f}, max={int(tiles_passed.max())}")
    print(f"[RESULT] Reached 16s: {int(reached_16.sum())}/{count}")
    print(f"[RESULT] Advanced >=1m after 16s: {int(progressed_after_16.sum())}/{int(reached_16.sum())}")
    if bool(reached_16.any()):
        print(f"[RESULT] Additional frontier after 16s (16s survivors): mean={post_16[reached_16].mean().item():.3f} m")
    print(f"[RESULT] Survived full {env_cfg.episode_length_s:.0f}s: {int(full_time.sum())}/{count}")
    print(f"[RESULT] Full-time survivors with >=0.10m new frontier in final 2s: {int(moving_at_end.sum())}/{int(full_time.sum())}")
    for name in names:
        print(f"[RESULT] End reason {name}: {int(terminal_flags[name].sum())}/{count}")
    for index, name in enumerate(first_names):
        mask = shape_ids == index
        if bool(mask.any()):
            print(f"[RESULT] start={name}: full_time={int((full_time & mask).sum())}/{int(mask.sum())}, grounded_frontier_mean={grounded_max_x[mask].mean().item():.3f} m")

    if args_cli.output_csv:
        path = Path(args_cli.output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "env_id", "start_shape", "reward", "steps", "seconds", "end_reasons", "full_time",
                "final_x_m", "max_x_m", "grounded_max_x_m", "tiles_passed",
                "reached_16s", "additional_frontier_after_16s_m", "moving_in_final_2s",
                *[f"grounded_frontier_at_{seconds:g}s_m" for seconds in marks],
            ])
            for index in range(count):
                writer.writerow([
                    index, first_names[int(shape_ids[index])], reward[index].item(), int(steps[index]),
                    duration[index].item(), "+".join(name for name in names if terminal_flags[name][index]) or "unfinished",
                    int(full_time[index]), final_x[index].item(), max_x[index].item(), grounded_max_x[index].item(),
                    int(tiles_passed[index]), int(reached_16[index]), post_16[index].item(), int(moving_at_end[index]),
                    *[snapshot[index].item() for _, snapshot in snapshots.values()],
                ])
        print(f"[INFO] Per-environment results saved to: {path}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
