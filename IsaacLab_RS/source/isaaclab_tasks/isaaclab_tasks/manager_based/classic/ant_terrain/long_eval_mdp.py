"""Capture progress before automatic reset, without terminating the episode."""

import torch

from isaaclab.managers import ManagerTermBase


class RecordLongCourseProgress(ManagerTermBase):
    """Diagnostic-only term; all returned termination flags are always false."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.max_x = torch.zeros(env.num_envs, device=env.device)
        self.grounded_max_x = torch.zeros_like(self.max_x)
        self.last_x = torch.zeros_like(self.max_x)
        self.last_max_x = torch.zeros_like(self.max_x)
        self.last_grounded_max_x = torch.zeros_like(self.max_x)
        self.never_terminate = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.max_x[env_ids] = 0.0
        self.grounded_max_x[env_ids] = 0.0
        # Keep last_* intact: env.step() auto-resets before returning to the
        # evaluator, which still needs the terminal step's actual measurements.

    def __call__(self, env):
        x = env.scene["robot"].data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
        contacts = env.scene["foot_contacts"]
        grounded = (contacts.data.net_forces_w.norm(dim=-1) > contacts.cfg.force_threshold).any(dim=1)
        finite = torch.isfinite(x)
        self.max_x[:] = torch.maximum(self.max_x, torch.where(finite, x, self.max_x))
        self.grounded_max_x[:] = torch.maximum(
            self.grounded_max_x, torch.where(finite & grounded, x, self.grounded_max_x),
        )
        self.last_x.copy_(x)
        self.last_max_x.copy_(self.max_x)
        self.last_grounded_max_x.copy_(self.grounded_max_x)
        return self.never_terminate
