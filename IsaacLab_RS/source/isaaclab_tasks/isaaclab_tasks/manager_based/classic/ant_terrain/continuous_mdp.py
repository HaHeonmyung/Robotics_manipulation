"""Forward walking and reset-time curriculum for long mixed courses."""

import torch


def forward_progress(env, speed_cap: float = 1.0):
    """Reward contact-supported +X motion, without rewarding ever faster flight."""
    speed = env.scene["robot"].data.root_lin_vel_w[:, 0].clamp(-speed_cap, speed_cap)
    contact = env.scene["foot_contacts"].data.net_forces_w.norm(dim=-1).amax(dim=1) > 1.0
    return speed * contact.float()


def corridor_out_of_bounds(env, min_x: float = -3.7, max_x: float = 43.7, half_width: float = 1.0):
    local_xy = env.scene["robot"].data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    return (local_xy[:, 0] < min_x) | (local_xy[:, 0] > max_x) | (local_xy[:, 1].abs() > half_width)


def continuous_terrain_levels(env, env_ids, advance_distance: float = 6.0, retreat_distance: float = 3.3):
    """Advance after a full, safe episode with progress; retain late-failure levels.

    The separate start counter excludes artificial early timeouts introduced by
    RSL-RL's random initial episode lengths. A first reset never changes levels.
    """
    terrain = env.scene.terrain
    if not hasattr(env, "_continuous_episode_start_step"):
        env._continuous_episode_start_step = torch.full(
            (env.num_envs,), env.common_step_counter, dtype=torch.long, device=env.device,
        )
    elapsed_steps = env.common_step_counter - env._continuous_episode_start_step[env_ids]
    full_episode = elapsed_steps >= env.max_episode_length
    active_episode = env.episode_length_buf[env_ids] > 0
    distance = env.scene["robot"].data.root_pos_w[env_ids, 0] - env.scene.env_origins[env_ids, 0]
    failed = env.termination_manager.terminated[env_ids]
    timed_out = env.termination_manager.get_term("time_out")[env_ids]
    move_up = active_episode & full_episode & timed_out & ~failed & (distance >= advance_distance)
    move_down = active_episode & (failed | full_episode) & (distance < retreat_distance) & ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    env._continuous_episode_start_step[env_ids] = env.common_step_counter
    return terrain.terrain_levels.float().mean()
