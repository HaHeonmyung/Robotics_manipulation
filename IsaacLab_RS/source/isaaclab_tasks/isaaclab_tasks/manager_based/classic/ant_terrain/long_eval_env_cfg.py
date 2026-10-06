"""Sixty-second, 24-tile evaluation for the existing Recovery checkpoint."""

from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from . import continuous_mdp
from .long_eval_mdp import RecordLongCourseProgress
from .long_eval_terrain import AntLongMixedCourseTerrainCfg
from .recovery_env_cfg import AntContinuousWideRayRecoveryEvalEnvCfg


@configclass
class AntContinuousWideRayRecoveryLongEvalEnvCfg(AntContinuousWideRayRecoveryEvalEnvCfg):
    """Same sensor, dynamics, scoring and failure rules; extend time and course."""

    def __post_init__(self):
        super().__post_init__()
        generator = self.scene.terrain.terrain_generator
        generator.size = (192.0, 8.0)
        generator.sub_terrains = {
            name: AntLongMixedCourseTerrainCfg(
                proportion=cfg.proportion, first_shape=name, held_out=True,
            )
            for name, cfg in generator.sub_terrains.items()
        }
        self.episode_length_s = 60.0
        self.terminations.corridor_out_of_bounds = DoneTerm(
            func=continuous_mdp.corridor_out_of_bounds,
            params={"min_x": -3.7, "max_x": 187.7, "half_width": 1.0},
        )
        self.terminations.long_course_progress = DoneTerm(func=RecordLongCourseProgress)
