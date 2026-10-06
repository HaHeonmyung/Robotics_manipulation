"""Reward new forward ground rather than oscillation, and discourage long stalls."""

import math

import torch

from isaaclab.managers import ManagerTermBase


def _local_x(env):
    return env.scene["robot"].data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]


def _contact(env):
    sensor = env.scene["foot_contacts"]
    return (sensor.data.net_forces_w.norm(dim=-1) > sensor.cfg.force_threshold).any(dim=1)


class GroundedFrontierProgress(ManagerTermBase):
    """Only newly visited +X distance earns reward; a short retreat earns zero."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.best_x = _local_x(env).clone()

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.best_x[env_ids] = _local_x(self._env)[env_ids]

    def __call__(self, env, speed_cap: float = 1.0):
        x = _local_x(env)
        new_distance = (x - self.best_x).clamp_min(0.0)
        # Update even during flight so landing cannot collect distance flown.
        self.best_x[:] = torch.maximum(self.best_x, x)
        return (new_distance / env.step_dt).clamp_max(speed_cap) * _contact(env).float()


class StuckWindowPenalty(ManagerTermBase):
    """A reset-safe rolling window of the episode's furthest forward position."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.window_steps = math.ceil(cfg.params.get("window_s", 2.0) / env.step_dt)
        if self.window_steps < 1:
            raise ValueError("The stuck window must contain at least one policy step.")
        self.best_x = _local_x(env).clone()
        self.history = self.best_x.repeat(self.window_steps + 1, 1)
        self.start_step = torch.full(
            (env.num_envs,), env.common_step_counter, dtype=torch.long, device=env.device,
        )
        self.has_landed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.best_x[env_ids] = _local_x(self._env)[env_ids]
        self.history[:, env_ids] = self.best_x[env_ids].unsqueeze(0)
        self.start_step[env_ids] = self._env.common_step_counter
        self.has_landed[env_ids] = False

    def __call__(self, env, window_s: float = 2.0, minimum_advance_m: float = 0.10):
        self.best_x[:] = torch.maximum(self.best_x, _local_x(env))
        self.has_landed |= _contact(env)
        size = self.window_steps + 1
        previous_slot = (env.common_step_counter - self.window_steps) % size
        advance = self.best_x - self.history[previous_slot]
        self.history[env.common_step_counter % size] = self.best_x
        complete_window = env.common_step_counter - self.start_step >= self.window_steps
        self.last_advance_m = advance.clone()
        self.last_stuck = complete_window & self.has_landed & (advance < minimum_advance_m)
        return self.last_stuck.float()
