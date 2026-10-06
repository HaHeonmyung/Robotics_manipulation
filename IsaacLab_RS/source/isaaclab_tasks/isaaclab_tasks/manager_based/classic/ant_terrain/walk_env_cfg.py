"""Corrected locomotion experiment; keep the earlier hard tasks reproducible."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import MySceneCfg, ObservationsCfg, RewardsCfg

from .hard_env_cfg import AntHardRayPolicyCfg, AntTerrainHardEnvCfg, _set_held_out_terrain
from . import walk_mdp


@configclass
class AntWalkSceneCfg(MySceneCfg):
    # Shared environment instrumentation. Neither policy receives this probe.
    ground_probe = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(0.0, 0.0)),
        mesh_prim_paths=["/World/ground"],
    )
    foot_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_foot", history_length=3, track_air_time=True,
    )


@configclass
class AntWalkRewardsCfg(RewardsCfg):
    progress = RewTerm(
        func=walk_mdp.grounded_progress, weight=1.0,
        params={"target_pos": (1000.0, 0.0, 0.0)},
    )
    prolonged_flight = RewTerm(func=walk_mdp.prolonged_flight, weight=-2.0)
    vertical_speed = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)


@configclass
class AntWalkHardEnvCfg(AntTerrainHardEnvCfg):
    scene: AntWalkSceneCfg = AntWalkSceneCfg(num_envs=4096, env_spacing=5.0, clone_in_fabric=False)
    rewards: AntWalkRewardsCfg = AntWalkRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.activate_contact_sensors = True
        # ±1 policy commands map to ±7.5 Nm, with a second explicit motor limit.
        self.actions.joint_effort.clip = {".*": (-7.5, 7.5)}
        self.scene.robot.actuators["body"].effort_limit_sim = 7.5
        self.scene.ground_probe.update_period = self.decimation * self.sim.dt
        self.terminations.torso_height = DoneTerm(
            func=walk_mdp.height_below_ground, params={"minimum_height": 0.31},
        )
        self.terminations.tilt = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.2})
        self.terminations.flight = DoneTerm(func=walk_mdp.extended_flight, params={"limit_s": 0.35})


@configclass
class AntWalkRaySceneCfg(AntWalkSceneCfg):
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=(1.6, 1.2)),
        mesh_prim_paths=["/World/ground"],
    )


@configclass
class AntWalkRayObservationsCfg(ObservationsCfg):
    policy: AntHardRayPolicyCfg = AntHardRayPolicyCfg()


@configclass
class AntWalkHardRayEnvCfg(AntWalkHardEnvCfg):
    scene: AntWalkRaySceneCfg = AntWalkRaySceneCfg(num_envs=4096, env_spacing=5.0, clone_in_fabric=False)
    observations: AntWalkRayObservationsCfg = AntWalkRayObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt


@configclass
class AntWalkHardEvalEnvCfg(AntWalkHardEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _set_held_out_terrain(self)


@configclass
class AntWalkHardRayEvalEnvCfg(AntWalkHardRayEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _set_held_out_terrain(self)


@configclass
class AntWalkHardScoreEnvCfg(AntWalkHardEvalEnvCfg):
    """Same held-out walking task, scored with the unmodified Ant reward."""

    rewards: RewardsCfg = RewardsCfg()


@configclass
class AntWalkHardRayScoreEnvCfg(AntWalkHardRayEvalEnvCfg):
    """Keep the 123 inputs expected by the model; use the original Ant reward."""

    rewards: RewardsCfg = RewardsCfg()


def _set_bounded_course(cfg):
    # One row preserves the requested difficulty range; fixed columns identify shapes.
    generator = cfg.scene.terrain.terrain_generator
    generator.curriculum = True
    generator.num_rows = 1
    cfg.terminations.course_out_of_bounds = DoneTerm(
        func=walk_mdp.course_out_of_bounds, params={"half_length": 3.7},
    )
    # The stairs end at |x|=3 m; finish on the landing inside the 8 m tile.
    cfg.terminations.course_goal = DoneTerm(
        func=walk_mdp.course_goal_reached, params={"distance": 3.3, "half_length": 3.7},
    )


@configclass
class AntWalkCourseHardEvalEnvCfg(AntWalkHardEvalEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _set_bounded_course(self)


@configclass
class AntWalkCourseHardRayEvalEnvCfg(AntWalkHardRayEvalEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _set_bounded_course(self)


@configclass
class AntCourseCurriculumCfg:
    terrain_levels = CurrTerm(func=walk_mdp.course_terrain_levels)


def _set_course_curriculum(cfg):
    _set_bounded_course(cfg)
    generator = cfg.scene.terrain.terrain_generator
    generator.num_rows = 10
    generator.difficulty_range = (0.0, 1.0)
    cfg.scene.terrain.max_init_terrain_level = 0
    # Standing still must not earn a sustained survival or heading reward.
    cfg.rewards.alive.weight = 0.0
    cfg.rewards.move_to_target.weight = 0.0
    cfg.rewards.progress.weight = 5.0
    cfg.rewards.energy.weight = -0.01
    cfg.rewards.course_goal = RewTerm(
        func=walk_mdp.course_goal_reached, weight=100.0, params={"distance": 3.3, "half_length": 3.7},
    )


@configclass
class AntWalkCourseCurriculumEnvCfg(AntWalkHardEnvCfg):
    curriculum: AntCourseCurriculumCfg = AntCourseCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        _set_course_curriculum(self)


@configclass
class AntWalkCourseCurriculumRayEnvCfg(AntWalkHardRayEnvCfg):
    curriculum: AntCourseCurriculumCfg = AntCourseCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        _set_course_curriculum(self)
