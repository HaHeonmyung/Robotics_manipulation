# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Toy2Box variant that adds object positions relative to the end effector (coursework axis-C experiment).

Scene, actions, events, terminations and success term are identical to
RmpFlowAgibotPlaceToy2BoxEnvCfg; only two observation terms are added.
"""

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.place import mdp as place_mdp
from isaaclab_tasks.manager_based.manipulation.stack import mdp

from .place_toy2box_rmp_rel_env_cfg import RmpFlowAgibotPlaceToy2BoxEnvCfg


def object_rel_to_eef(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg) -> torch.Tensor:
    """Object position minus end-effector position, both expressed in the robot base frame."""
    obj = place_mdp.object_poses_in_base_frame(env, object_cfg=object_cfg, return_key="pos")
    ee = mdp.ee_frame_pose_in_base_frame(env, return_key="pos")
    return obj - ee


@configclass
class RmpFlowAgibotPlaceToy2BoxRelObsEnvCfg(RmpFlowAgibotPlaceToy2BoxEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.toy_truck_rel = ObsTerm(
            func=object_rel_to_eef, params={"object_cfg": SceneEntityCfg("toy_truck")}
        )
        self.observations.policy.box_rel = ObsTerm(func=object_rel_to_eef, params={"object_cfg": SceneEntityCfg("box")})
