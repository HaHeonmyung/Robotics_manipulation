"""Ant dynamics perturbations; observation/reward/action dimensions stay unchanged."""
import json
from pathlib import Path
from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg, EventCfg

PROTOCOL = json.loads(Path(__file__).with_name('protocol.json').read_text())

@configclass
class DynamicsEvents(EventCfg):
    material: EventTermCfg | None = None
    mass: EventTermCfg | None = None
    kick: EventTermCfg | None = None

def parameters(condition):
    if condition == 'baseline' or condition == 'nominal': return PROTOCOL['nominal']
    if condition == 'randomized': return PROTOCOL['randomized_training']
    return PROTOCOL['held_out'][condition]

def make_config(condition, num_envs, seed, device):
    p = parameters(condition)
    cfg = AntEnvCfg()
    cfg.seed = seed
    cfg.scene.num_envs = num_envs
    cfg.sim.device = device
    cfg.scene.terrain.physics_material.friction_combine_mode = 'multiply'
    cfg.sim.physics_material.friction_combine_mode = 'multiply'
    cfg.events = DynamicsEvents()
    cfg.events.material = EventTermCfg(func=mdp.randomize_rigid_body_material, mode='startup', params={
        'asset_cfg': SceneEntityCfg('robot', body_names='.*'),
        'static_friction_range': tuple(p['friction']), 'dynamic_friction_range': tuple(p['friction']),
        'restitution_range': (0.0, 0.0), 'num_buckets': 64, 'make_consistent': True,
    })
    cfg.events.mass = EventTermCfg(func=mdp.randomize_rigid_body_mass, mode='startup', params={
        'asset_cfg': SceneEntityCfg('robot', body_names='.*'),
        'mass_distribution_params': tuple(p['mass_scale']), 'operation': 'scale', 'recompute_inertia': True,
    })
    if p['kick']:
        k = p['kick']
        cfg.events.kick = EventTermCfg(func=mdp.push_by_setting_velocity, mode='interval',
            interval_range_s=tuple(p['kick_interval_s']), params={
                'asset_cfg': SceneEntityCfg('robot'), 'velocity_range': {'x': (-k,k), 'y': (-k,k)}})
    cfg.viewer.origin_type = 'asset_root'
    cfg.viewer.asset_name = 'robot'
    cfg.viewer.env_index = 0
    cfg.viewer.eye = (-4.0, 4.0, 2.5)
    cfg.viewer.lookat = (0.0, 0.0, 0.5)
    cfg.viewer.resolution = (640, 480)
    return cfg
