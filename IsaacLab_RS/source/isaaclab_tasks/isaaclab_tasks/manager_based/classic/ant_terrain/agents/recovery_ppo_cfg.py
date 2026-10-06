"""Use the earlier PPO defaults instead of the unbounded entropy experiment."""

from isaaclab.utils import configclass

from .wide_ray_ppo_cfg import AntContinuousWideRayPPORunnerCfg


@configclass
class AntContinuousWideRayRecoveryPPORunnerCfg(AntContinuousWideRayPPORunnerCfg):
    experiment_name = "ant_continuous_wide_ray_recovery"


@configclass
class AntContinuousWideRayStuckPPORunnerCfg(AntContinuousWideRayRecoveryPPORunnerCfg):
    experiment_name = "ant_continuous_wide_ray_stuck"
