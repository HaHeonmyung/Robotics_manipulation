"""Separate fresh-training experiment with an exploratory entropy bonus."""

from isaaclab.utils import configclass

from .wide_ray_ppo_cfg import AntContinuousWideRayPPORunnerCfg


@configclass
class AntContinuousWideRayLandingPPORunnerCfg(AntContinuousWideRayPPORunnerCfg):
    experiment_name = "ant_continuous_wide_ray_landing"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.entropy_coef = 0.005
