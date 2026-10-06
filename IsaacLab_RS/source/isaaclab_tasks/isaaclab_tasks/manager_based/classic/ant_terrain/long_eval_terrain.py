"""Extend the held-out six-shape course without changing the original tasks."""

import numpy as np

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass

from .continuous_terrain import AntMixedCourseTerrainCfg, mixed_course_terrain


def long_mixed_course_terrain(difficulty: float, cfg):
    """Concatenate independently shuffled six-tile courses at zero-height seams."""
    shapes = list(ROUGH_TERRAINS_CFG.sub_terrains)
    block_length = cfg.tile_size * len(shapes)
    expected_size = (block_length * cfg.num_blocks, cfg.tile_size)
    if cfg.num_blocks < 1 or cfg.size != expected_size:
        raise ValueError(f"Long course requires positive blocks and size {expected_size}, got {cfg.size}")
    meshes, spawn_origin = [], None
    for index in range(cfg.num_blocks):
        # Generate each block before drawing the next one: the first block uses
        # the same random draws as an ordinary six-tile course for a given state.
        first_shape = cfg.first_shape if index == 0 else str(np.random.choice(shapes))
        block = AntMixedCourseTerrainCfg(
            size=(block_length, cfg.tile_size), tile_size=cfg.tile_size,
            first_shape=first_shape, held_out=cfg.held_out,
        )
        block_meshes, origin = mixed_course_terrain(difficulty, block)
        offset = np.array([index * block_length, 0.0, 0.0])
        for mesh in block_meshes:
            mesh.apply_translation(offset)
        meshes.extend(block_meshes)
        if index == 0:
            spawn_origin = np.asarray(origin, dtype=float)
    return meshes, spawn_origin


@configclass
class AntLongMixedCourseTerrainCfg(SubTerrainBaseCfg):
    function = long_mixed_course_terrain
    size = (192.0, 8.0)
    tile_size: float = 8.0
    num_blocks: int = 4
    first_shape: str = "pyramid_stairs"
    held_out: bool = True
