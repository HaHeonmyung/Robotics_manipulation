"""Matched high-difficulty Ant tasks with and without a height scanner."""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import MySceneCfg, ObservationsCfg

from .ant_terrain_env_cfg import AntTerrainEnvCfg


@configclass
class AntTerrainHardEnvCfg(AntTerrainEnvCfg):
    """Ant on the stock rough-terrain shapes at top-row difficulty (0.9 to 1.0)."""

    def __post_init__(self):
        super().__post_init__()
        # RayCaster counts sensor parents in USD; Fabric-only clones leave only env_0 there.
        self.scene.clone_in_fabric = False
        generator = ROUGH_TERRAINS_CFG.copy()
        generator.curriculum = False
        generator.difficulty_range = (0.9, 1.0)
        self.scene.terrain.terrain_generator = generator


@configclass
class AntHardRaySceneCfg(MySceneCfg):
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=(1.6, 1.2)),
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )


@configclass
class AntHardRayPolicyCfg(ObservationsCfg.PolicyCfg):
    height_scan = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        clip=(-1.0, 1.0),
    )


@configclass
class AntHardRayObservationsCfg(ObservationsCfg):
    policy: AntHardRayPolicyCfg = AntHardRayPolicyCfg()


@configclass
class AntTerrainHardRayEnvCfg(AntTerrainHardEnvCfg):
    """Same hard terrain and Ant MDP, plus 63 local height samples."""

    scene: AntHardRaySceneCfg = AntHardRaySceneCfg(num_envs=4096, env_spacing=5.0, clone_in_fabric=False)
    observations: AntHardRayObservationsCfg = AntHardRayObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt


def _set_held_out_terrain(cfg):
    """Change only terrain geometry parameters and friction for evaluation."""
    terrain = cfg.scene.terrain
    sub_terrains = terrain.terrain_generator.sub_terrains
    sub_terrains["pyramid_stairs"].step_width = 0.28
    sub_terrains["pyramid_stairs_inv"].step_width = 0.28
    sub_terrains["boxes"].grid_width = 0.43
    sub_terrains["random_rough"].noise_step = 0.025
    terrain.physics_material.static_friction = 0.8
    terrain.physics_material.dynamic_friction = 0.8


@configclass
class AntTerrainHardEvalEnvCfg(AntTerrainHardEnvCfg):
    """Held-out terrain for evaluating the Ant without height scans."""

    def __post_init__(self):
        super().__post_init__()
        _set_held_out_terrain(self)


@configclass
class AntTerrainHardRayEvalEnvCfg(AntTerrainHardRayEnvCfg):
    """The same held-out terrain, with the trained height scanner present."""

    def __post_init__(self):
        super().__post_init__()
        _set_held_out_terrain(self)
