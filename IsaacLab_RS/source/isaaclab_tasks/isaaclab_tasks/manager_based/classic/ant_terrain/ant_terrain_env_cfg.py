"""Start the Ant on a gentle mix of generated terrains."""

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg

from . import terrain_mdp


@configclass
class AntTerrainEnvCfg(AntEnvCfg):
    """Ant task with smaller rough-terrain obstacles for initial training."""

    def __post_init__(self):
        super().__post_init__()

        terrain = self.scene.terrain
        terrain.terrain_type = "generator"
        terrain.terrain_generator = ROUGH_TERRAINS_CFG.copy()
        terrain.terrain_generator.curriculum = False

        # Start below the stock rough-terrain difficulty. Raise these ranges later.
        sub_terrains = terrain.terrain_generator.sub_terrains
        sub_terrains["pyramid_stairs"].step_height_range = (0.02, 0.07)
        sub_terrains["pyramid_stairs_inv"].step_height_range = (0.02, 0.07)
        sub_terrains["boxes"].grid_height_range = (0.02, 0.08)
        sub_terrains["random_rough"].noise_range = (0.01, 0.04)
        sub_terrains["random_rough"].noise_step = 0.01
        sub_terrains["hf_pyramid_slope"].slope_range = (0.0, 0.15)
        sub_terrains["hf_pyramid_slope_inv"].slope_range = (0.0, 0.15)

        # The stock Ant uses world Z. Inverted terrain can start below world Z=0.
        self.observations.policy.base_height.func = terrain_mdp.base_height_above_origin
        self.terminations.torso_height.func = terrain_mdp.root_height_below_origin
