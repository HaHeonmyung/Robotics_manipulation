"""Reuse the Ant PPO settings with a separate experiment directory."""

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.classic.ant.agents.rsl_rl_ppo_cfg import AntPPORunnerCfg


@configclass
class AntTerrainPPORunnerCfg(AntPPORunnerCfg):
    experiment_name = "ant_terrain"


@configclass
class AntHardPPORunnerCfg(AntPPORunnerCfg):
    experiment_name = "ant_terrain_hard"


@configclass
class AntHardRayPPORunnerCfg(AntPPORunnerCfg):
    experiment_name = "ant_terrain_hard_ray"


@configclass
class AntWalkPPORunnerCfg(AntPPORunnerCfg):
    experiment_name = "ant_walk_hard"
    clip_actions = 1.0


@configclass
class AntWalkRayPPORunnerCfg(AntWalkPPORunnerCfg):
    experiment_name = "ant_walk_hard_ray"


@configclass
class AntCoursePPORunnerCfg(AntWalkPPORunnerCfg):
    experiment_name = "ant_course_curriculum"


@configclass
class AntCourseRayPPORunnerCfg(AntCoursePPORunnerCfg):
    experiment_name = "ant_course_curriculum_ray"


@configclass
class AntContinuousPPORunnerCfg(AntWalkPPORunnerCfg):
    experiment_name = "ant_continuous"


@configclass
class AntContinuousRayPPORunnerCfg(AntContinuousPPORunnerCfg):
    experiment_name = "ant_continuous_ray"
