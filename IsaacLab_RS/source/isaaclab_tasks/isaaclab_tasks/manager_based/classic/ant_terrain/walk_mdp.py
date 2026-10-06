"""Ground-relative termination and contact-aware walking rewards."""

import torch


def height_below_ground(env, minimum_height: float):
    ground_z = env.scene["ground_probe"].data.ray_hits_w[:, 0, 2]
    height = env.scene["robot"].data.root_pos_w[:, 2] - ground_z
    return ~torch.isfinite(height) | (height < minimum_height)


def _all_feet_air_time(env):
    return env.scene["foot_contacts"].data.current_air_time.min(dim=1).values


def prolonged_flight(env, grace_s: float = 0.15):
    return (_all_feet_air_time(env) > grace_s).float()


def extended_flight(env, limit_s: float):
    return _all_feet_air_time(env) > limit_s


def grounded_progress(env, target_pos: tuple[float, float, float]):
    robot = env.scene["robot"]
    direction = torch.tensor(target_pos, device=env.device)[:2] - robot.data.root_pos_w[:, :2]
    direction /= direction.norm(dim=1, keepdim=True).clamp_min(1.0e-6)
    speed = (robot.data.root_lin_vel_w[:, :2] * direction).sum(dim=1)
    contact = env.scene["foot_contacts"].data.net_forces_w.norm(dim=-1).amax(dim=1) > 1.0
    return speed * contact.float()


def course_out_of_bounds(env, half_length: float = 3.0, half_width: float = 1.0):
    local_xy = env.scene["robot"].data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    return (local_xy[:, 0].abs() > half_length) | (local_xy[:, 1].abs() > half_width)


def course_goal_reached(env, distance: float = 2.7, half_length: float = 3.0, half_width: float = 1.0):
    """Local target reached upright with foot contact and within the course."""
    robot = env.scene["robot"]
    local_x = robot.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    contact = env.scene["foot_contacts"].data.net_forces_w.norm(dim=-1).amax(dim=1) > 1.0
    upright = robot.data.projected_gravity_b[:, 2] < -0.36235775  # cos(1.2 rad)
    return (
        (local_x >= distance)
        & ~course_out_of_bounds(env, half_length, half_width)
        & ~height_below_ground(env, 0.31)
        & upright
        & contact
        & ~extended_flight(env, 0.35)
    )


def course_terrain_levels(env, env_ids):
    terrain = env.scene.terrain
    if env.common_step_counter > 0:
        passed = env.termination_manager.get_term("course_goal")[env_ids]
        terrain.update_env_origins(env_ids, passed, ~passed)
    return terrain.terrain_levels.float().mean()
