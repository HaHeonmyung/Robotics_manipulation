"""Inspect the training Ant RayCaster on a plane, with live grid values.

Usage from the Isaac Lab root:
    ./isaaclab.sh -p scripts/demos/sensors/ant_height_map.py
    ./isaaclab.sh -p scripts/demos/sensors/ant_height_map.py --headless --steps 120

The pose is held for sensor inspection. No learned policy is needed.
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="One Ant on a plane with its training height scanner.")
parser.add_argument("--steps", type=int, default=0, help="Physics steps before exiting; 0 runs until the GUI closes.")
parser.add_argument("--torso_height", type=float, default=0.5, help="Held torso height above the plane, in meters.")
parser.add_argument("--yaw", type=float, default=0.0, help="Held torso yaw, in degrees.")
parser.add_argument("--wide", action="store_true", help="Inspect the 187-ray, forward-shifted training scanner.")
parser.add_argument("--output_dir", default=None, help="Save final heatmap, arrays and metadata here.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps < 0 or not 0.31 <= args.torso_height <= 2.0:
    parser.error("steps must be nonnegative and height must be between 0.31 and 2.0 meters")
if args.headless and args.steps == 0:
    args.steps = 120
if args.output_dir is None:
    args.output_dir = "logs/ant_raycaster_wide_debug" if args.wide else "logs/ant_raycaster_debug"

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import json
import faulthandler
import math
import signal
import time
from types import SimpleNamespace

if hasattr(signal, "SIGUSR1"):
    faulthandler.register(signal.SIGUSR1, all_threads=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab_tasks.manager_based.classic.ant_terrain.walk_env_cfg import AntWalkRaySceneCfg
from isaaclab_tasks.manager_based.classic.ant_terrain.wide_ray_env_cfg import AntWideRaySceneCfg


def grid_indices(sensor):
    """Map the actual ray ordering to rows (+X forward) and columns (+Y left)."""
    xy = np.round(sensor.ray_starts[0, :, :2].cpu().numpy(), 6)
    x_values = np.unique(xy[:, 0])[::-1]
    y_values = np.unique(xy[:, 1])[::-1]
    indices = np.full((len(x_values), len(y_values)), -1, dtype=int)
    for ray_id, (x, y) in enumerate(xy):
        indices[np.flatnonzero(x_values == x)[0], np.flatnonzero(y_values == y)[0]] = ray_id
    if np.any(indices < 0):
        raise RuntimeError("The scanner pattern is not a complete rectangular grid")
    return indices, x_values, y_values


def cell_color(value):
    if not np.isfinite(value):
        return 0xFF777777
    r, g, b, a = (int(c * 255) for c in matplotlib.colormaps["coolwarm"](np.clip(value + 0.5, 0.0, 1.0)))
    return r | (g << 8) | (b << 16) | (a << 24)


class HeightMapWindow:
    def __init__(self, indices, x_values, y_values):
        import omni.ui as ui
        self.ui = ui
        self.indices = indices
        self.rectangles = []
        self.labels = []
        self.window = ui.Window("Ant RayCaster - live height map", width=850 if args.wide else 590,
                                height=1000 if args.wide else 790)
        with self.window.frame:
            with ui.VStack(spacing=6):
                description = ("187 rays | 3.2 x 2.0 m | spacing 0.2 m | forward offset 0.6 m" if args.wide
                               else "63 rays | 1.6 x 1.2 m | spacing 0.2 m")
                ui.Label(description, height=24)
                ui.Label("Pink dots in the viewport = ray hit points", height=22)
                ui.Label("Pose is held. Move the sliders to inspect the sensor.", height=22)
                controls = (("Torso Z (m)", args.torso_height, 0.31, 2.0),
                            ("Yaw (degrees)", args.yaw, -180.0, 180.0),
                            ("World X (m)", 0.0, -2.0, 2.0),
                            ("World Y (m)", 0.0, -2.0, 2.0))
                self.models = []
                for title, value, minimum, maximum in controls:
                    with ui.HStack(height=24):
                        ui.Label(title, width=130)
                        slider = ui.FloatSlider(min=minimum, max=maximum)
                        slider.model.set_value(value)
                        self.models.append(slider.model)
                self.mode = ui.ComboBox(0, "Ground Z (m)", "Policy height input (m)", height=28)
                self.stats = ui.Label("Waiting for ray hits...", height=24)
                ui.Label("Top = forward +X | left = +Y | right = -Y", height=24)
                with ui.HStack(height=24):
                    ui.Spacer(width=65)
                    for y in y_values:
                        ui.Label(f"{y:+.1f}", width=68, alignment=ui.Alignment.CENTER)
                for row, x in enumerate(x_values):
                    with ui.HStack(height=31 if args.wide else 39, spacing=2):
                        ui.Label(f"X {x:+.1f}", width=63)
                        for col in range(len(y_values)):
                            with ui.ZStack(width=66, height=29 if args.wide else 37):
                                rectangle = ui.Rectangle(style={"background_color": cell_color(0.0)})
                                label = ui.Label("---", alignment=ui.Alignment.CENTER,
                                                 style={"color": 0xFF202020, "font_size": 14})
                            self.rectangles.append(rectangle)
                            self.labels.append(label)
                ui.Label("Cell values are meters. Colors use a fixed -0.5 to +0.5 m range.", height=22)
                ui.Label("Policy input = torso Z - ground Z - 0.5, clipped to [-1, 1].", height=22)
                ui.Label("On a plane, Ground Z should be near 0 everywhere.", height=22)
                ui.Label("Higher ground gives a LOWER policy input at a fixed torso height.", height=22)

    def pose(self):
        return [model.get_value_as_float() for model in self.models]

    def update(self, ground_z, policy_height, sensor_z):
        mode = self.mode.model.get_item_value_model().get_value_as_int()
        values = (ground_z if mode == 0 else policy_height)[self.indices].ravel()
        finite = values[np.isfinite(values)]
        self.stats.text = (f"Valid {len(finite)}/{len(values)} | torso Z {sensor_z:.3f} m | "
                           f"min {finite.min():+.3f}, max {finite.max():+.3f}" if len(finite)
                           else "No valid ray hits")
        for rectangle, label, value in zip(self.rectangles, self.labels, values):
            rectangle.style = {"background_color": cell_color(value)}
            label.text = f"{value:+.3f}" if np.isfinite(value) else "MISS"


def save_snapshot(output_dir, hits, policy_height, sensor_z, indices, x_values, y_values):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ground_grid = hits[:, 2][indices]
    policy_grid = policy_height[indices]
    fig, axes = plt.subplots(1, 2, figsize=(11, 7), constrained_layout=True)
    for axis, values, title in zip(axes, (ground_grid, policy_grid), ("Ground Z (m)", "Policy height input (m)")):
        picture = axis.imshow(values, vmin=-0.5, vmax=0.5, cmap="coolwarm")
        axis.set_title(title)
        axis.set_xticks(np.arange(len(y_values)), [f"{v:+.1f}" for v in y_values])
        axis.set_yticks(np.arange(len(x_values)), [f"{v:+.1f}" for v in x_values])
        axis.set_xlabel("Lateral Y (m): +Y left, -Y right")
        axis.set_ylabel("Forward X (m): +X at top")
        for row, col in np.ndindex(values.shape):
            value = values[row, col]
            axis.text(col, row, f"{value:.3f}" if np.isfinite(value) else "MISS", ha="center", va="center", fontsize=8)
        fig.colorbar(picture, ax=axis, shrink=0.7, label="m")
    fig.suptitle(f"Ant training RayCaster | {len(hits)} rays | torso Z = {sensor_z:.3f} m")
    fig.savefig(output_dir / "height_map.png", dpi=160)
    plt.close(fig)
    np.savez(output_dir / "height_map.npz", ray_hits_w=hits, ground_z=ground_grid,
             policy_height=policy_grid, grid_ray_indices=indices, x=x_values, y=y_values)
    finite = np.isfinite(hits).all(axis=1)
    metadata = {"num_rays": len(hits), "valid_hits": int(finite.sum()), "grid_shape": list(indices.shape),
                "sensor_z_m": sensor_z, "ground_z_min_m": float(hits[finite, 2].min()) if finite.any() else None,
                "ground_z_max_m": float(hits[finite, 2].max()) if finite.any() else None,
                "policy_height_min_m": float(policy_height[finite].min()) if finite.any() else None,
                "policy_height_max_m": float(policy_height[finite].max()) if finite.any() else None,
                "mode": "held_pose_on_plane", "scanner_profile": "wide" if args.wide else "narrow",
                "x_range_body_m": [float(x_values.min()), float(x_values.max())],
                "y_range_body_m": [float(y_values.min()), float(y_values.max())],
                "policy_height_formula": "sensor_z - ground_z - 0.5; clip [-1,1]"}
    (output_dir / "height_map.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[RESULT] {json.dumps(metadata)}", flush=True)
    print(f"[INFO] Saved heatmap and sensor arrays to {output_dir.resolve()}", flush=True)


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, render_interval=2, device=args.device))
    # This scene class has a plane by default, and the exact scanner used in training.
    scene_cls = AntWideRaySceneCfg if args.wide else AntWalkRaySceneCfg
    cfg = scene_cls(num_envs=1, env_spacing=5.0, clone_in_fabric=False)
    cfg.robot.spawn.activate_contact_sensors = True
    cfg.height_scanner.debug_vis = True
    cfg.height_scanner.update_period = 1.0 / 60.0
    cfg.height_scanner.visualizer_cfg.markers["hit"].radius = 0.018
    cfg.height_scanner.visualizer_cfg.markers["hit"].visual_material.diffuse_color = (1.0, 0.1, 0.6)
    scene = InteractiveScene(cfg)
    sim.reset()
    sim.set_camera_view([2.7, 2.7, 2.6], [0.0, 0.0, 0.1])
    robot = scene["robot"]
    sensor = scene["height_scanner"]
    sensor.update(sim.get_physics_dt(), force_recompute=True)
    indices, x_values, y_values = grid_indices(sensor)
    expected_rays, expected_shape = (187, (17, 11)) if args.wide else (63, (9, 7))
    if sensor.num_rays != expected_rays or indices.shape != expected_shape:
        raise RuntimeError(f"Unexpected training scanner layout: {sensor.num_rays}, {indices.shape}")
    panel = HeightMapWindow(indices, x_values, y_values) if not args.headless else None
    fake_env = SimpleNamespace(scene=scene)
    entity_cfg = SceneEntityCfg("height_scanner")
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    joints = robot.data.default_joint_pos.clone()
    zeros = torch.zeros_like(robot.data.default_joint_vel)
    step = 0
    snapshot = None
    print(f"[INFO] One Ant, plane, held pose, {sensor.num_rays} ray hits. No checkpoint required.", flush=True)
    try:
        with torch.inference_mode():
            while simulation_app.is_running() and (not args.steps or step < args.steps):
                started = time.perf_counter()
                height, yaw, x, y = panel.pose() if panel else (args.torso_height, args.yaw, 0.0, 0.0)
                half_yaw = math.radians(yaw) / 2.0
                root_state[:, 0] = scene.env_origins[:, 0] + x
                root_state[:, 1] = scene.env_origins[:, 1] + y
                root_state[:, 2] = height
                root_state[:, 3:7] = torch.tensor([math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)], device=sim.device)
                robot.write_root_pose_to_sim(root_state[:, :7])
                robot.write_root_velocity_to_sim(torch.zeros_like(root_state[:, 7:]))
                robot.write_joint_state_to_sim(joints, zeros)
                robot.set_joint_effort_target(torch.zeros_like(joints))
                scene.write_data_to_sim()
                sim.step()
                scene.update(sim.get_physics_dt())
                sensor.update(0.0, force_recompute=True)
                step += 1
                if step == 1 or step % 8 == 0 or step == args.steps:
                    hits = sensor.data.ray_hits_w[0].cpu().numpy().copy()
                    policy_height = mdp.height_scan(fake_env, entity_cfg)[0].clamp(-1.0, 1.0).cpu().numpy().copy()
                    sensor_z = sensor.data.pos_w[0, 2].item()
                    snapshot = (hits, policy_height, sensor_z)
                    if panel:
                        panel.update(hits[:, 2], policy_height, sensor_z)
                    if step == 1 or step % 120 == 0:
                        finite = np.isfinite(hits).all(axis=1)
                        print(f"[SCAN] valid={int(finite.sum())}/{sensor.num_rays} | torso Z={sensor_z:.3f} m | "
                              f"ground Z range={hits[finite, 2].min():+.5f}..{hits[finite, 2].max():+.5f} m", flush=True)
                if not args.headless:
                    remaining = sim.get_physics_dt() - (time.perf_counter() - started)
                    if remaining > 0:
                        time.sleep(remaining)
    finally:
        if snapshot is not None:
            save_snapshot(args.output_dir, *snapshot, indices, x_values, y_values)
        if panel:
            panel.window.destroy()
        # Unsubscribe the standalone stop callback before closing the application.
        sim.clear_all_callbacks()
        sim.clear_instance()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close(wait_for_replicator=False)
