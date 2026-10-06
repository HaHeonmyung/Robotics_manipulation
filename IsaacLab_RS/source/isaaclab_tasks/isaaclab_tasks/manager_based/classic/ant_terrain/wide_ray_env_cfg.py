"""Matched Ant environments with a wider, forward-shifted height scanner."""

from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from .continuous_env_cfg import AntContinuousRayEnvCfg, AntContinuousRayEvalEnvCfg
from .walk_env_cfg import AntWalkHardRayScoreEnvCfg, AntWalkRaySceneCfg


@configclass
class AntWideRaySceneCfg(AntWalkRaySceneCfg):
    """187 downward rays: x=-1.0..2.2 m and y=-1.0..1.0 m in body yaw."""

    height_scanner = AntWalkRaySceneCfg(num_envs=1, env_spacing=5.0).height_scanner.replace(
        offset=RayCasterCfg.OffsetCfg(pos=(0.6, 0.0, 20.0)),
        pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=(3.2, 2.0)),
    )


@configclass
class AntContinuousWideRayEnvCfg(AntContinuousRayEnvCfg):
    """Same continuous curriculum, dynamics, rewards and terminations; 247 inputs."""

    scene: AntWideRaySceneCfg = AntWideRaySceneCfg(num_envs=4096, env_spacing=5.0, clone_in_fabric=False)


@configclass
class AntContinuousWideRayEvalEnvCfg(AntContinuousRayEvalEnvCfg):
    """Same bounded held-out continuous course and original Ant scoring reward."""

    scene: AntWideRaySceneCfg = AntWideRaySceneCfg(num_envs=4096, env_spacing=5.0, clone_in_fabric=False)


@configclass
class AntWalkHardWideRayScoreEnvCfg(AntWalkHardRayScoreEnvCfg):
    """Same Score-Eval used for model_10949.pt, with the wide sensor inputs."""

    scene: AntWideRaySceneCfg = AntWideRaySceneCfg(num_envs=4096, env_spacing=5.0, clone_in_fabric=False)
