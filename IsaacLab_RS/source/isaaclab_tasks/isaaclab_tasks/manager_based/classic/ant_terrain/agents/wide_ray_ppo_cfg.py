"""Fresh wide-scan experiment with the same PPO settings as the narrow scan."""

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import AntContinuousRayPPORunnerCfg


@configclass
class AntContinuousWideRayPPORunnerCfg(AntContinuousRayPPORunnerCfg):
    experiment_name = "ant_continuous_wide_ray"
    max_iterations = 10000
