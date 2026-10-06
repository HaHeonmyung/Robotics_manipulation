"""Join stock 8 m terrain tiles into mixed courses at a common difficulty."""

import numpy as np

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass


def mixed_course_terrain(difficulty: float, cfg):
    """Six stock tiles, with zero-height borders meeting without a height offset.

    Spawn at the first tile's center, as in the earlier Ant course. Remaining
    shapes are shuffled, so stepping off the first tile leads to more obstacles.
    """
    shapes = list(ROUGH_TERRAINS_CFG.sub_terrains)
    if cfg.first_shape not in shapes:
        raise ValueError(f"Unknown first terrain: {cfg.first_shape}")
    expected_size = (cfg.tile_size * len(shapes), cfg.tile_size)
    if cfg.size != expected_size:
        raise ValueError(f"Mixed course must have size {expected_size}, got {cfg.size}")
    remainder = [shape for shape in shapes if shape != cfg.first_shape]
    sequence = [cfg.first_shape, *np.random.permutation(remainder).tolist()]
    meshes, spawn_origin = [], None
    for index, shape in enumerate(sequence):
        tile = ROUGH_TERRAINS_CFG.sub_terrains[shape].copy()
        tile.size = (cfg.tile_size, cfg.tile_size)
        if shape.startswith("hf_") or shape == "random_rough":
            tile.horizontal_scale = 0.1
            tile.vertical_scale = 0.005
            tile.slope_threshold = 0.75
        if shape == "random_rough":
            # The stock generator ignores difficulty: explicitly scale its noise.
            scale = 0.25 + 0.75 * difficulty
            tile.noise_range = tuple(value * scale for value in tile.noise_range)
            tile.noise_step = 0.005
        if cfg.held_out:
            if shape in {"pyramid_stairs", "pyramid_stairs_inv"}:
                tile.step_width = 0.28
            elif shape == "boxes":
                tile.grid_width = 0.43
            elif shape == "random_rough":
                tile.noise_step = 0.025
        tile_meshes, origin = tile.function(difficulty, tile)
        offset = np.array([index * cfg.tile_size, 0.0, 0.0])
        for mesh in tile_meshes:
            mesh.apply_translation(offset)
        meshes.extend(tile_meshes)
        if index == 0:
            spawn_origin = np.asarray(origin, dtype=float) + offset
    return meshes, spawn_origin


@configclass
class AntMixedCourseTerrainCfg(SubTerrainBaseCfg):
    function = mixed_course_terrain
    size = (48.0, 8.0)
    tile_size: float = 8.0
    first_shape: str = "pyramid_stairs"
    held_out: bool = False
