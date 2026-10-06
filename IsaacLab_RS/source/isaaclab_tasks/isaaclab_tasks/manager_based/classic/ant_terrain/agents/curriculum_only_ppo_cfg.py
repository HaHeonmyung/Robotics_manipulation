"""Use the exact hard-baseline PPO settings with a separate experiment name."""

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import AntHardPPORunnerCfg


@configclass
class AntTerrainCurriculumOnlyPPORunnerCfg(AntHardPPORunnerCfg):
    experiment_name = "ant_terrain_curriculum"
