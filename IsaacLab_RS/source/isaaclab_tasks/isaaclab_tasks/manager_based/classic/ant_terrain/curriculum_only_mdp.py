"""Reset-time difficulty updates for the otherwise unchanged hard Ant task."""

import torch


def terrain_levels_only(env, env_ids, advance_distance: float = 3.3):
    """Promote after a full safe episode; demote after failure or stalled timeout.

    RSL-RL randomizes the first episode-length buffer. An independent clock
    prevents those shortened timeouts from being treated as full episodes.
    """
    if not hasattr(env, "_terrain_only_episode_start_step"):
        env._terrain_only_episode_start_step = torch.full(
            (env.num_envs,), env.common_step_counter, dtype=torch.long, device=env.device,
        )
    elapsed = env.common_step_counter - env._terrain_only_episode_start_step[env_ids]
    active = env.episode_length_buf[env_ids] > 0
    full_episode = elapsed >= env.max_episode_length
    failed = env.termination_manager.terminated[env_ids]
    timed_out = env.termination_manager.get_term("time_out")[env_ids]
    distance = env.scene["robot"].data.root_pos_w[env_ids, 0] - env.scene.env_origins[env_ids, 0]
    move_up = active & full_episode & timed_out & ~failed & (distance >= advance_distance)
    move_down = active & (failed | (full_episode & timed_out & ~move_up))
    env.scene.terrain.update_env_origins(env_ids, move_up, move_down)
    env._terrain_only_episode_start_step[env_ids] = env.common_step_counter
    return env.scene.terrain.terrain_levels.float().mean()
