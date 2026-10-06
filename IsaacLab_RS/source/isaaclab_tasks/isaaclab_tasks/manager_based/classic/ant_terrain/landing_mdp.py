"""Allow a bounded first landing without relaxing flight limits after contact."""

import torch

from isaaclab.managers import ManagerTermBase


class FlightAfterInitialLanding(ManagerTermBase):
    """Use an independent reset clock, including RSL-RL's randomized first episodes."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.has_landed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.start_step = torch.full(
            (env.num_envs,), env.common_step_counter, dtype=torch.long, device=env.device,
        )

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.has_landed[env_ids] = False
        self.start_step[env_ids] = self._env.common_step_counter

    def __call__(self, env, limit_s: float = 0.35, initial_landing_s: float = 1.0):
        contacts = env.scene["foot_contacts"]
        force = contacts.data.net_forces_w.norm(dim=-1)
        contact_now = (force > contacts.cfg.force_threshold).any(dim=1)
        self.has_landed |= contact_now
        air_time = contacts.data.current_air_time.min(dim=1).values
        elapsed_s = (env.common_step_counter - self.start_step) * env.step_dt
        # Copies preserve terminal measurements after the environment auto-resets.
        self.last_has_landed = self.has_landed.clone()
        self.last_air_time_s = air_time.clone()
        self.last_elapsed_s = elapsed_s.clone()
        return (
            (self.has_landed & (air_time > limit_s))
            | (~self.has_landed & (elapsed_s >= initial_landing_s))
        )
