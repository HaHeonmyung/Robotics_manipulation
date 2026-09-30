# Source provenance

- IsaacLab: https://github.com/isaac-sim/IsaacLab
- Tag: v2.3.0
- Commit: 3c6e67bb5c7ada942a6d1884ab69338f57596f77
- The tracked source tree is included under `IsaacLab_RS/`, with its original licenses intact.
- Course evaluation helper `scripts/reinforcement_learning/rsl_rl/play_one_episode.py`: from https://github.com/cailab-hy/IsaacLab_RS, local main commit e83a5d2f11ca1b5f03b690e1978479e620c500e2. Only its final shutdown call was changed to `close(skip_cleanup=True)` to avoid shutdown hanging on this laptop.
- New experiment code and outputs: `IsaacLab_RS/coursework/ant-generalization/`.
- Isaac Sim assets and Python/Conda environments are not redistributed. Install the environment described in the experiment README before running.
