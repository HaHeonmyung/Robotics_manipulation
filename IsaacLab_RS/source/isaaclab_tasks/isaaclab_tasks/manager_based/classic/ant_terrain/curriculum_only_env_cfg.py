"""Experiment 3: the sensor-free hard Ant baseline plus terrain curriculum."""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from .curriculum_only_mdp import terrain_levels_only
from .hard_env_cfg import AntTerrainHardEnvCfg


@configclass
class AntTerrainOnlyCurriculumCfg:
    terrain_levels = CurrTerm(func=terrain_levels_only, params={"advance_distance": 3.3})


@configclass
class AntTerrainCurriculumOnlyEnvCfg(AntTerrainHardEnvCfg):
    """Keep baseline observations, rewards, dynamics and terminations unchanged."""

    curriculum: AntTerrainOnlyCurriculumCfg = AntTerrainOnlyCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        generator = self.scene.terrain.terrain_generator
        generator.curriculum = True
        generator.difficulty_range = (0.0, 1.0)
        self.scene.terrain.max_init_terrain_level = 0
