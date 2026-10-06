"""Matched evaluation worlds: only policy height-map observations differ."""

from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import MySceneCfg

from .common_eval_mdp import CommonEvalRayCaster, CommonEvalTerrainImporter, RecordCommonCourseProgress
from .continuous_mdp import corridor_out_of_bounds
from .continuous_terrain import AntMixedCourseTerrainCfg
from .hard_env_cfg import AntHardRayObservationsCfg, AntTerrainHardEnvCfg
from .long_eval_terrain import AntLongMixedCourseTerrainCfg


@configclass
class AntCommonEvalSceneCfg(MySceneCfg):
    # Passive diagnostic sensor shared by ALL policies; not a policy input.
    foot_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_foot", history_length=3, track_air_time=True,
    )


@configclass
class AntCommonNarrowEvalSceneCfg(AntCommonEvalSceneCfg):
    height_scanner = RayCasterCfg(
        class_type=CommonEvalRayCaster,
        prim_path="{ENV_REGEX_NS}/Robot/torso",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=(1.6, 1.2)),
        mesh_prim_paths=["/World/ground"],
    )


@configclass
class AntCommonWideEvalSceneCfg(AntCommonNarrowEvalSceneCfg):
    height_scanner = AntCommonNarrowEvalSceneCfg(num_envs=1, env_spacing=5.0).height_scanner.replace(
        offset=RayCasterCfg.OffsetCfg(pos=(0.6, 0.0, 20.0)),
        pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=(3.2, 2.0)),
    )


@configclass
class AntCommonEvalEnvCfg(AntTerrainHardEnvCfg):
    """Experiment-2 dynamics/rewards/height rule on a bounded held-out course.

    Root height is relative to the initial tile origin, exactly as experiment 2.
    All policies use the same original, unclipped Ant effort action settings.
    There are no tilt/flight/goal/stuck terminations or training-shaped rewards.
    """

    scene: AntCommonEvalSceneCfg = AntCommonEvalSceneCfg(num_envs=100, env_spacing=5.0, clone_in_fabric=False)

    def __post_init__(self):
        super().__post_init__()
        self.seed = 24
        self.curriculum = None
        self.scene.robot.spawn.activate_contact_sensors = True
        terrain = self.scene.terrain
        terrain.class_type = CommonEvalTerrainImporter
        proportions = {
            "pyramid_stairs": 0.2, "pyramid_stairs_inv": 0.2,
            "boxes": 0.2, "random_rough": 0.2,
            "hf_pyramid_slope": 0.1, "hf_pyramid_slope_inv": 0.1,
        }
        terrain.terrain_generator = TerrainGeneratorCfg(
            seed=24, size=(48.0, 8.0), border_width=1.0, num_rows=1, num_cols=10,
            curriculum=True, use_cache=False, difficulty_range=(0.9, 1.0),
            sub_terrains={
                name: AntMixedCourseTerrainCfg(proportion=ratio, first_shape=name, held_out=True)
                for name, ratio in proportions.items()
            },
        )
        # Generator curriculum only arranges columns; no RL difficulty updates.
        terrain.max_init_terrain_level = 0
        terrain.physics_material.static_friction = 0.8
        terrain.physics_material.dynamic_friction = 0.8
        self.episode_length_s = 16.0
        self.terminations.corridor_out_of_bounds = DoneTerm(
            func=corridor_out_of_bounds, params={"min_x": -3.7, "max_x": 43.7, "half_width": 1.0},
        )
        self.terminations.long_course_progress = DoneTerm(func=RecordCommonCourseProgress)
        if hasattr(self.scene, "height_scanner"):
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt


@configclass
class AntCommonNarrowEvalEnvCfg(AntCommonEvalEnvCfg):
    scene: AntCommonNarrowEvalSceneCfg = AntCommonNarrowEvalSceneCfg(num_envs=100, env_spacing=5.0, clone_in_fabric=False)
    observations: AntHardRayObservationsCfg = AntHardRayObservationsCfg()


@configclass
class AntCommonWideEvalEnvCfg(AntCommonNarrowEvalEnvCfg):
    scene: AntCommonWideEvalSceneCfg = AntCommonWideEvalSceneCfg(num_envs=100, env_spacing=5.0, clone_in_fabric=False)


def _extend_common_course(cfg):
    generator = cfg.scene.terrain.terrain_generator
    generator.size = (192.0, 8.0)
    generator.sub_terrains = {
        name: AntLongMixedCourseTerrainCfg(proportion=sub.proportion, first_shape=name, held_out=True)
        for name, sub in generator.sub_terrains.items()
    }
    cfg.episode_length_s = 60.0
    cfg.terminations.corridor_out_of_bounds.params["max_x"] = 187.7


@configclass
class AntCommonLongEvalEnvCfg(AntCommonEvalEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _extend_common_course(self)


@configclass
class AntCommonNarrowLongEvalEnvCfg(AntCommonNarrowEvalEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _extend_common_course(self)


@configclass
class AntCommonWideLongEvalEnvCfg(AntCommonWideEvalEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _extend_common_course(self)
