"""Use one identical inference wrapper/architecture for every comparison task."""

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import AntHardPPORunnerCfg


@configclass
class AntCommonEvalPPORunnerCfg(AntHardPPORunnerCfg):
    experiment_name = "ant_common_eval"
