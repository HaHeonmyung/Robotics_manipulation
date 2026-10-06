"""Continuous mixed-terrain tasks; preserve the original observation layout."""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import RewardsCfg

from . import continuous_mdp
from .continuous_terrain import AntMixedCourseTerrainCfg
from .walk_env_cfg import AntWalkHardEnvCfg, AntWalkHardRayEnvCfg


@configclass
class AntContinuousCurriculumCfg:
    terrain_levels = CurrTerm(func=continuous_mdp.continuous_terrain_levels)


def _set_continuous_course(cfg, evaluation: bool = False):
    proportions = {
        "pyramid_stairs": 0.2, "pyramid_stairs_inv": 0.2,
        "boxes": 0.2, "random_rough": 0.2,
        "hf_pyramid_slope": 0.1, "hf_pyramid_slope_inv": 0.1,
    }
    cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
        size=(48.0, 8.0), border_width=1.0, num_rows=1 if evaluation else 10,
        num_cols=10, curriculum=True, use_cache=False,
        difficulty_range=(0.9, 1.0) if evaluation else (0.0, 1.0),
        sub_terrains={
            name: AntMixedCourseTerrainCfg(proportion=ratio, first_shape=name, held_out=evaluation)
            for name, ratio in proportions.items()
        },
    )
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.episode_length_s = 16.0
    cfg.terminations.corridor_out_of_bounds = DoneTerm(func=continuous_mdp.corridor_out_of_bounds)
    # No goal termination/reward: the policy keeps walking beyond the old 3.3 m target.
    if not evaluation:
        cfg.rewards.alive.weight = 0.0
        cfg.rewards.move_to_target.weight = 0.0
        cfg.rewards.energy.weight = -0.01
        cfg.rewards.progress = RewTerm(func=continuous_mdp.forward_progress, weight=5.0)
    else:
        cfg.scene.terrain.physics_material.static_friction = 0.8
        cfg.scene.terrain.physics_material.dynamic_friction = 0.8


@configclass
class AntContinuousEnvCfg(AntWalkHardEnvCfg):
    curriculum: AntContinuousCurriculumCfg = AntContinuousCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        _set_continuous_course(self)


@configclass
class AntContinuousRayEnvCfg(AntWalkHardRayEnvCfg):
    curriculum: AntContinuousCurriculumCfg = AntContinuousCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        _set_continuous_course(self)


@configclass
class AntContinuousEvalEnvCfg(AntWalkHardEnvCfg):
    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        _set_continuous_course(self, evaluation=True)


@configclass
class AntContinuousRayEvalEnvCfg(AntWalkHardRayEnvCfg):
    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        _set_continuous_course(self, evaluation=True)
