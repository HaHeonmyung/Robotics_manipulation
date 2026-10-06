"""Evaluate checkpoints with matched terrain, physics and original Ant rewards."""

import argparse
import csv
import gc
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", action="append", required=True, help="Repeat to compare multiple policies.")
parser.add_argument("--label", action="append", help="Optional label for each checkpoint, in the same order.")
parser.add_argument("--seed", type=int, default=24)
parser.add_argument("--num_envs", type=int, default=100)
parser.add_argument("--duration", type=int, choices=(16, 60), default=16)
parser.add_argument("--disable_torso_height", action="store_true")
parser.add_argument("--output_dir", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
paths = [Path(p).expanduser().resolve() for p in args.checkpoint]
if args.label is not None and len(args.label) != len(paths):
    parser.error("Provide one --label per --checkpoint, or omit all labels.")
for path in paths:
    if not path.is_file():
        parser.error(f"Checkpoint does not exist: {path}")
labels = args.label or [f"{p.parent.name}_{p.stem}" for p in paths]
labels = [re.sub(r"[^\w.-]", "_", label) for label in labels]
if len(set(labels)) != len(labels):
    parser.error("Checkpoint labels must be unique.")
height_mode = "off" if args.disable_torso_height else "on"
output = args.output_dir or Path(
    f"logs/ant_terrain_comparison/common_{args.duration}s_height_{height_mode}_seed{args.seed}/"
    + datetime.now().strftime("%Y%m%d_%H%M%S")
)
output.mkdir(parents=True, exist_ok=True)


def write_summary(rows, destination):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with destination.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


if len(paths) > 1:
    # Isaac Sim stage teardown/recreation in one process can stall. Give each
    # model a fresh simulator process, then verify and combine its measurements.
    forwarded = []
    argv = iter(sys.argv[1:])
    for token in argv:
        if token in ("--checkpoint", "--label", "--output_dir"):
            next(argv)
        elif not token.startswith(("--checkpoint=", "--label=", "--output_dir=")):
            forwarded.append(token)
    rows = []
    for path, label in zip(paths, labels):
        child_dir = output / label
        child_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = child_dir / f"{label}.json"
        metadata_path.unlink(missing_ok=True)
        print(f"[BATCH] Evaluating {label} in a fresh simulator process.", flush=True)
        subprocess.run([
            sys.executable, "-u", str(Path(__file__).resolve()), *forwarded,
            "--checkpoint", str(path), "--label", label, "--output_dir", str(child_dir),
        ], check=True)
        summary = json.loads(metadata_path.read_text())["summary"]
        if rows:
            for key in ("shared_config_sha256", "terrain_mesh_sha256", "initial_state_sha256"):
                if summary[key] != rows[0][key]:
                    raise RuntimeError(f"{key} differs between models; comparison is not matched.")
        rows.append(summary)
        write_summary(rows, output / "summary.csv")
    print(f"[INFO] Matched settings, terrain and initial states verified for {len(rows)} model(s).", flush=True)
    print(f"[INFO] Summary CSV: {output / 'summary.csv'}", flush=True)
    sys.exit(0)

simulation_app = AppLauncher(args).app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.classic.ant_terrain import common_eval_env_cfg as configs
from isaaclab_tasks.manager_based.classic.ant_terrain.agents.common_eval_ppo_cfg import AntCommonEvalPPORunnerCfg


def serializable(value):
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(v) for v in value]
    if callable(value):
        return f"{value.__module__}:{value.__qualname__}"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def fingerprint(value):
    return hashlib.sha256(json.dumps(serializable(value), sort_keys=True).encode()).hexdigest()


def shared_configuration(cfg):
    shared = cfg.to_dict()
    shared["scene"].pop("height_scanner", None)
    shared["observations"]["policy"].pop("height_scan", None)
    return shared


# Check all six registered profiles, including variants not selected this run.
for group in (
    (configs.AntCommonEvalEnvCfg, configs.AntCommonNarrowEvalEnvCfg, configs.AntCommonWideEvalEnvCfg),
    (configs.AntCommonLongEvalEnvCfg, configs.AntCommonNarrowLongEvalEnvCfg, configs.AntCommonWideLongEvalEnvCfg),
):
    signatures = {fingerprint(shared_configuration(cls())) for cls in group}
    if len(signatures) != 1:
        raise RuntimeError("Common task profiles differ beyond height-map observations.")


def evaluate(path, label, expected_signature):
    state = torch.load(path, map_location="cpu", weights_only=False)
    obs_dim = state["model_state_dict"]["actor.0.weight"].shape[1]
    checkpoint_iter = state.get("iter")
    del state
    options = {
        60: ("", configs.AntCommonEvalEnvCfg, configs.AntCommonLongEvalEnvCfg),
        123: ("-Ray", configs.AntCommonNarrowEvalEnvCfg, configs.AntCommonNarrowLongEvalEnvCfg),
        247: ("-Wide-Ray", configs.AntCommonWideEvalEnvCfg, configs.AntCommonWideLongEvalEnvCfg),
    }
    if obs_dim not in options:
        raise ValueError(f"Unsupported policy input size {obs_dim}; expected 60, 123 or 247.")
    suffix, short_cfg, long_cfg = options[obs_dim]
    task = f"Isaac-Ant-Common{suffix}{'-Long' if args.duration == 60 else ''}-Eval-v0"
    cfg = (short_cfg if args.duration == 16 else long_cfg)()
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.scene.terrain.terrain_generator.seed = args.seed
    if args.device:
        cfg.sim.device = args.device
    if args.disable_torso_height:
        cfg.terminations.torso_height = None
    shared_signature = fingerprint(shared_configuration(cfg))
    if expected_signature is not None and shared_signature != expected_signature:
        raise RuntimeError("Evaluation settings differ beyond the permitted height-map observation fields.")
    print(f"[MODEL] {label}: inputs={obs_dim}, task={task}, checkpoint_iter={checkpoint_iter}", flush=True)
    agent = AntCommonEvalPPORunnerCfg()
    agent.seed = args.seed
    agent.device = cfg.sim.device
    env = RslRlVecEnvWrapper(gym.make(task, cfg=cfg), clip_actions=agent.clip_actions)
    try:
        base = env.unwrapped
        assert base.observation_manager.group_obs_dim["policy"] == (obs_dim,)
        runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=agent.device)
        runner.load(str(path))
        policy = runner.get_inference_policy(device=base.device)
        # Reset after network construction under an identical RNG state, so
        # different input dimensions cannot change the sampled initial joints.
        base.seed(args.seed)
        obs, _ = env.reset()
        robot = base.scene["robot"]
        initial_state = hashlib.sha256()
        initial_values = {
            "root_state": robot.data.root_state_w.detach().cpu().clone(),
            "joint_pos": robot.data.joint_pos.detach().cpu().clone(),
            "joint_vel": robot.data.joint_vel.detach().cpu().clone(),
            "env_origins": base.scene.env_origins.detach().cpu().clone(),
        }
        for tensor in initial_values.values():
            initial_state.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        initial_signature = initial_state.hexdigest()
        mesh_signature = base.scene.terrain.mesh_sha256
        manager = base.termination_manager
        tracker = manager.get_term_cfg("long_course_progress").func
        names = [name for name in manager.active_terms if name != "long_course_progress"]
        n, device = env.num_envs, base.device
        rewards = torch.zeros(n, dtype=torch.float64, device=device)
        steps = torch.zeros(n, dtype=torch.long, device=device)
        finished = torch.zeros(n, dtype=torch.bool, device=device)
        final_x = torch.zeros(n, device=device)
        raw_max = torch.zeros_like(final_x)
        grounded_max = torch.zeros_like(final_x)
        air_steps = torch.zeros(n, dtype=torch.long, device=device)
        flags = {name: torch.zeros_like(finished) for name in names}
        with torch.inference_mode():
            for _ in range(base.max_episode_length):
                active = ~finished
                obs, reward, dones, _ = env.step(policy(obs))
                rewards[active] += reward[active]
                steps[active] += 1
                final_x[active] = tracker.last_x[active]
                raw_max[active] = tracker.last_max_x[active]
                grounded_max[active] = tracker.last_grounded_max_x[active]
                air_steps[active] = tracker.last_air_steps[active]
                ended = active & dones.bool()
                for name in names:
                    flags[name][ended] = manager.get_term(name)[ended]
                finished |= ended
                if bool(finished.all()):
                    break
        if not bool(finished.all()):
            raise RuntimeError("Not all first episodes completed; refusing to report partial results.")
        seconds = steps.double() * base.step_dt
        air_fraction = air_steps.double() / steps.clamp_min(1)
        tiles = ((grounded_max + 4.0) / 8.0).floor().clamp_min(0).long()
        failed = torch.zeros_like(finished)
        for name in names:
            if not manager.get_term_cfg(name).time_out:
                failed |= flags[name]
        full_time = flags["time_out"] & ~failed
        generator = cfg.scene.terrain.terrain_generator
        shape_names = list(generator.sub_terrains)
        cumulative = torch.tensor([s.proportion for s in generator.sub_terrains.values()], device=device)
        cumulative = (cumulative / cumulative.sum()).cumsum(0)
        shape_ids = torch.searchsorted(cumulative, base.scene.terrain.terrain_types.float() / generator.num_cols + 0.001)
        csv_path = output / f"{label}.csv"
        with csv_path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["env_id", "start_shape", "reward", "steps", "seconds", "end_reasons", "full_time", "final_x_m", "raw_max_x_m", "grounded_max_x_m", "tiles_passed", "air_fraction"])
            for i in range(n):
                writer.writerow([
                    i, shape_names[int(shape_ids[i])], rewards[i].item(), int(steps[i]), seconds[i].item(),
                    "+".join(name for name in names if flags[name][i]), int(full_time[i]), final_x[i].item(),
                    raw_max[i].item(), grounded_max[i].item(), int(tiles[i]), air_fraction[i].item(),
                ])
        summary = {
            "label": label, "checkpoint": str(path), "checkpoint_iter": checkpoint_iter,
            "task": task, "obs_dim": obs_dim, "seed": args.seed, "num_envs": n,
            "duration_limit_s": args.duration, "torso_height_enabled": not args.disable_torso_height,
            "reward_mean": rewards.mean().item(), "reward_std": rewards.std(unbiased=False).item(),
            "seconds_mean": seconds.mean().item(), "seconds_std": seconds.std(unbiased=False).item(),
            "grounded_distance_mean_m": grounded_max.mean().item(), "grounded_distance_std_m": grounded_max.std(unbiased=False).item(),
            "tiles_passed_mean": tiles.double().mean().item(), "tiles_passed_max": int(tiles.max()),
            "full_time_count": int(full_time.sum()), "air_fraction_mean": air_fraction.mean().item(),
            "shared_config_sha256": shared_signature, "terrain_mesh_sha256": mesh_signature,
            "initial_state_sha256": initial_signature,
            **{f"end_{name}_count": int(flags[name].sum()) for name in names},
        }
        metadata = {
            "summary": summary, "environment": serializable(cfg.to_dict()), "runner": serializable(agent.to_dict()),
            "initial_state": {key: tensor.tolist() for key, tensor in initial_values.items()},
        }
        (output / f"{label}.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n")
        print(f"[RESULT] {label}: reward={summary['reward_mean']:.6f} ± {summary['reward_std']:.6f}, grounded_distance={summary['grounded_distance_mean_m']:.3f} m, full_time={summary['full_time_count']}/{n}, air_fraction={summary['air_fraction_mean']:.3f}", flush=True)
        return summary
    finally:
        env.close()


if __name__ == "__main__":
    try:
        summaries = []
        reference = None
        for path, label in zip(paths, labels):
            summary = evaluate(path, label, reference)
            if summaries:
                for key in ("terrain_mesh_sha256", "initial_state_sha256"):
                    if summary[key] != summaries[0][key]:
                        raise RuntimeError(f"{key} differs between models; comparison is not matched.")
            reference = summary["shared_config_sha256"]
            summaries.append(summary)
            write_summary(summaries, output / "summary.csv")
            gc.collect()
        print(f"[INFO] Matched settings, terrain and initial states verified for {len(summaries)} model(s).", flush=True)
        print(f"[INFO] Summary CSV: {output / 'summary.csv'}", flush=True)
    finally:
        simulation_app.close()
