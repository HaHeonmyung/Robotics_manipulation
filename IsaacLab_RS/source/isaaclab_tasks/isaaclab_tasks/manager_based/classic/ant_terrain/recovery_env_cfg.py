"""Keep the previous tasks intact while isolating two reward experiments."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from . import continuous_mdp, recovery_mdp
from .landing_env_cfg import (
    AntContinuousWideRayLandingEnvCfg,
    AntContinuousWideRayLandingEvalEnvCfg,
    AntWalkHardWideRayLandingScoreEnvCfg,
)


@configclass
class AntContinuousWideRayRecoveryEnvCfg(AntContinuousWideRayLandingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.progress = RewTerm(
            func=recovery_mdp.GroundedFrontierProgress, weight=5.0,
            params={"speed_cap": 1.0},
        )
        self.rewards.stuck = RewTerm(
            func=recovery_mdp.StuckWindowPenalty, weight=-0.5,
            params={"window_s": 2.0, "minimum_advance_m": 0.10},
        )


@configclass
class AntContinuousWideRayStuckEnvCfg(AntContinuousWideRayRecoveryEnvCfg):
    """Control: add the same stuck term but retain the signed speed reward."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.progress = RewTerm(func=continuous_mdp.forward_progress, weight=5.0)


@configclass
class AntContinuousWideRayRecoveryEvalEnvCfg(AntContinuousWideRayLandingEvalEnvCfg):
    """Use the original Ant evaluation reward, without the training stuck term."""


@configclass
class AntWalkHardWideRayRecoveryScoreEnvCfg(AntWalkHardWideRayLandingScoreEnvCfg):
    """Original Ant Score-Eval reward with the established first-landing grace."""
