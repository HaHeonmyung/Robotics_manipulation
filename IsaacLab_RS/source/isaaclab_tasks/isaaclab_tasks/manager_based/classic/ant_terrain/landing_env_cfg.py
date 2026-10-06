"""Wide height maps with first-landing grace; preserve terrain and dynamics."""

from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from .landing_mdp import FlightAfterInitialLanding
from .wide_ray_env_cfg import (
    AntContinuousWideRayEnvCfg,
    AntContinuousWideRayEvalEnvCfg,
    AntWalkHardWideRayScoreEnvCfg,
)


def _allow_initial_landing(cfg):
    cfg.terminations.flight = DoneTerm(
        func=FlightAfterInitialLanding,
        params={"limit_s": 0.35, "initial_landing_s": 1.0},
    )


@configclass
class AntContinuousWideRayLandingEnvCfg(AntContinuousWideRayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _allow_initial_landing(self)


@configclass
class AntContinuousWideRayLandingEvalEnvCfg(AntContinuousWideRayEvalEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _allow_initial_landing(self)


@configclass
class AntWalkHardWideRayLandingScoreEnvCfg(AntWalkHardWideRayScoreEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _allow_initial_landing(self)
