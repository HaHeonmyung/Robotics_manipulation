"""Ant locomotion on generated terrain."""

import gymnasium as gym

from . import agents

for suffix, config_name in (
    ("", "AntCommonEvalEnvCfg"),
    ("-Ray", "AntCommonNarrowEvalEnvCfg"),
    ("-Wide-Ray", "AntCommonWideEvalEnvCfg"),
    ("-Long", "AntCommonLongEvalEnvCfg"),
    ("-Ray-Long", "AntCommonNarrowLongEvalEnvCfg"),
    ("-Wide-Ray-Long", "AntCommonWideLongEvalEnvCfg"),
):
    gym.register(
        id=f"Isaac-Ant-Common{suffix}-Eval-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.common_eval_env_cfg:{config_name}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.common_eval_ppo_cfg:AntCommonEvalPPORunnerCfg",
        },
    )

gym.register(
    id="Isaac-Ant-Terrain-Curriculum-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.curriculum_only_env_cfg:AntTerrainCurriculumOnlyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.curriculum_only_ppo_cfg:AntTerrainCurriculumOnlyPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Ant-Continuous-Wide-Ray-Recovery-Long-Eval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.long_eval_env_cfg:AntContinuousWideRayRecoveryLongEvalEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.recovery_ppo_cfg:AntContinuousWideRayRecoveryPPORunnerCfg",
    },
)

for task_id, config_name, runner in (
    ("Isaac-Ant-Continuous-Wide-Ray-Recovery-v0", "AntContinuousWideRayRecoveryEnvCfg", "AntContinuousWideRayRecoveryPPORunnerCfg"),
    ("Isaac-Ant-Continuous-Wide-Ray-Stuck-v0", "AntContinuousWideRayStuckEnvCfg", "AntContinuousWideRayStuckPPORunnerCfg"),
    ("Isaac-Ant-Continuous-Wide-Ray-Recovery-Eval-v0", "AntContinuousWideRayRecoveryEvalEnvCfg", "AntContinuousWideRayRecoveryPPORunnerCfg"),
    ("Isaac-Ant-Walk-Hard-Wide-Ray-Recovery-Score-Eval-v0", "AntWalkHardWideRayRecoveryScoreEnvCfg", "AntContinuousWideRayRecoveryPPORunnerCfg"),
):
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.recovery_env_cfg:{config_name}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.recovery_ppo_cfg:{runner}",
        },
    )

for task_id, config_name in (
    ("Isaac-Ant-Continuous-Wide-Ray-Landing-v0", "AntContinuousWideRayLandingEnvCfg"),
    ("Isaac-Ant-Continuous-Wide-Ray-Landing-Eval-v0", "AntContinuousWideRayLandingEvalEnvCfg"),
    ("Isaac-Ant-Walk-Hard-Wide-Ray-Landing-Score-Eval-v0", "AntWalkHardWideRayLandingScoreEnvCfg"),
):
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.landing_env_cfg:{config_name}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.landing_ppo_cfg:AntContinuousWideRayLandingPPORunnerCfg",
        },
    )

for task_id, config_name in (
    ("Isaac-Ant-Continuous-Wide-Ray-v0", "AntContinuousWideRayEnvCfg"),
    ("Isaac-Ant-Continuous-Wide-Ray-Eval-v0", "AntContinuousWideRayEvalEnvCfg"),
    ("Isaac-Ant-Walk-Hard-Wide-Ray-Score-Eval-v0", "AntWalkHardWideRayScoreEnvCfg"),
):
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.wide_ray_env_cfg:{config_name}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.wide_ray_ppo_cfg:AntContinuousWideRayPPORunnerCfg",
        },
    )

for suffix, config_name, agent_name in (
    ("", "AntContinuousEnvCfg", "AntContinuousPPORunnerCfg"),
    ("-Ray", "AntContinuousRayEnvCfg", "AntContinuousRayPPORunnerCfg"),
    ("-Eval", "AntContinuousEvalEnvCfg", "AntContinuousPPORunnerCfg"),
    ("-Ray-Eval", "AntContinuousRayEvalEnvCfg", "AntContinuousRayPPORunnerCfg"),
):
    gym.register(
        id=f"Isaac-Ant-Continuous{suffix}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.continuous_env_cfg:{config_name}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:{agent_name}",
        },
    )

for suffix, config_name, agent_name in (
    ("", "AntWalkCourseCurriculumEnvCfg", "AntCoursePPORunnerCfg"),
    ("-Ray", "AntWalkCourseCurriculumRayEnvCfg", "AntCourseRayPPORunnerCfg"),
):
    gym.register(
        id=f"Isaac-Ant-Course-Curriculum{suffix}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.walk_env_cfg:{config_name}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:{agent_name}",
        },
    )

for task_suffix, config_name, agent_name in (
    ("", "AntWalkHardEnvCfg", "AntWalkPPORunnerCfg"),
    ("-Ray", "AntWalkHardRayEnvCfg", "AntWalkRayPPORunnerCfg"),
    ("-Eval", "AntWalkHardEvalEnvCfg", "AntWalkPPORunnerCfg"),
    ("-Ray-Eval", "AntWalkHardRayEvalEnvCfg", "AntWalkRayPPORunnerCfg"),
    ("-Score-Eval", "AntWalkHardScoreEnvCfg", "AntWalkPPORunnerCfg"),
    ("-Ray-Score-Eval", "AntWalkHardRayScoreEnvCfg", "AntWalkRayPPORunnerCfg"),
    ("-Course-Eval", "AntWalkCourseHardEvalEnvCfg", "AntWalkPPORunnerCfg"),
    ("-Ray-Course-Eval", "AntWalkCourseHardRayEvalEnvCfg", "AntWalkRayPPORunnerCfg"),
):
    gym.register(
        id=f"Isaac-Ant-Walk-Hard{task_suffix}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.walk_env_cfg:{config_name}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:{agent_name}",
        },
    )

gym.register(
    id="Isaac-Ant-Terrain-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_terrain_env_cfg:AntTerrainEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntTerrainPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Ant-Terrain-Hard-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hard_env_cfg:AntTerrainHardEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntHardPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Ant-Terrain-Hard-Ray-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hard_env_cfg:AntTerrainHardRayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntHardRayPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Ant-Terrain-Hard-Eval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hard_env_cfg:AntTerrainHardEvalEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntHardPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Ant-Terrain-Hard-Ray-Eval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hard_env_cfg:AntTerrainHardRayEvalEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntHardRayPPORunnerCfg",
    },
)
