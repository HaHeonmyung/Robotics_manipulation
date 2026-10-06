"""Measurement-only instrumentation shared by all comparison policies."""

import hashlib

import torch

from isaaclab.terrains import TerrainImporter
from isaaclab.sensors import RayCaster

from .long_eval_mdp import RecordLongCourseProgress


class CommonEvalRayCaster(RayCaster):
    """Zero-drift scanning must not alter the robot reset's random draws.

    Stock RayCaster.reset draws drift samples even when all ranges are zero.
    Restore its RNG state so adding a height map cannot change initial joints.
    """

    def reset(self, env_ids=None):
        if self.cfg.drift_range != (0.0, 0.0) or any(
            tuple(value) != (0.0, 0.0) for value in self.cfg.ray_cast_drift_range.values()
        ):
            raise ValueError("Common evaluation requires deterministic zero sensor drift.")
        cuda = torch.device(self.device).type == "cuda"
        state = torch.cuda.get_rng_state(self.device) if cuda else torch.get_rng_state()
        try:
            super().reset(env_ids)
        finally:
            if cuda:
                torch.cuda.set_rng_state(state, self.device)
            else:
                torch.set_rng_state(state)


class CommonEvalTerrainImporter(TerrainImporter):
    """Keep a mesh fingerprint so comparisons verify the actual generated world."""

    def import_mesh(self, name, mesh):
        digest = hashlib.sha256()
        digest.update(mesh.vertices.tobytes())
        digest.update(mesh.faces.tobytes())
        self.mesh_sha256 = digest.hexdigest()
        return super().import_mesh(name, mesh)


class RecordCommonCourseProgress(RecordLongCourseProgress):
    """Measure terminal progress and airborne fraction before automatic reset."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.air_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.steps = torch.zeros_like(self.air_steps)
        self.last_air_steps = torch.zeros_like(self.air_steps)
        self.last_steps = torch.zeros_like(self.steps)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self.air_steps[env_ids] = 0
        self.steps[env_ids] = 0

    def __call__(self, env):
        result = super().__call__(env)
        contacts = env.scene["foot_contacts"]
        grounded = (contacts.data.net_forces_w.norm(dim=-1) > contacts.cfg.force_threshold).any(dim=1)
        self.air_steps += (~grounded).long()
        self.steps += 1
        self.last_air_steps.copy_(self.air_steps)
        self.last_steps.copy_(self.steps)
        return result
