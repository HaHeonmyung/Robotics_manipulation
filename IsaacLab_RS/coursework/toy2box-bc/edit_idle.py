"""Idle-segment edits for BC training data (controlled experiments B' and C).

Idle step: all six motion commands have |a| <= --threshold. Runs longer than
--keep steps are edited; runs touching the --protect steps around a gripper
switch are never edited.

Modes
  fill      B': keep every step; the surplus idle steps get the motion command
            of the last non-idle step before the run ("keep moving" label).
  approach  C : delete surplus idle steps only in the approach phase, i.e.
            earlier than --before-grasp steps before the first gripper close;
            everything from there on is untouched.

Usage:
    python edit_idle.py IN.hdf5 OUT.hdf5 --mode fill     [--keep 2] [--protect 10] [--threshold 0.03]
    python edit_idle.py IN.hdf5 OUT.hdf5 --mode approach [--keep 2] [--protect 10] [--threshold 0.03] [--before-grasp 40]
"""

import argparse

import h5py
import numpy as np

from trim_idle import cut_group


def idle_runs(actions, threshold):
    idle = (np.abs(actions[:, :6]) <= threshold).all(axis=1)
    x = np.r_[0, idle.astype(int), 0]
    starts = np.flatnonzero(np.diff(x) == 1)
    ends = np.flatnonzero(np.diff(x) == -1)
    return list(zip(starts, ends))


def protected_mask(actions, protect):
    n = len(actions)
    p = np.zeros(n, dtype=bool)
    for s in np.flatnonzero(np.diff(actions[:, 6]) != 0) + 1:
        p[max(0, s - protect) : min(n, s + protect + 1)] = True
    return p


def first_close(actions):
    g = actions[:, 6]
    for s in np.flatnonzero(np.diff(g) != 0) + 1:
        if g[s] < 0:
            return int(s)
    return len(actions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--mode", choices=["fill", "approach"], required=True)
    ap.add_argument("--keep", type=int, default=2)
    ap.add_argument("--protect", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=0.03)
    ap.add_argument("--before-grasp", type=int, default=40, help="approach mode: steps before first close that stay untouched")
    a = ap.parse_args()

    tot_in = tot_out = edited = 0
    with h5py.File(a.src, "r") as fi, h5py.File(a.dst, "w") as fo:
        go = fo.create_group("data")
        for k, v in fi["data"].attrs.items():
            go.attrs[k] = v
        for name in sorted(fi["data"], key=lambda s: int(s.split("_")[1])):
            demo = fi["data"][name]
            actions = demo["actions"][()]
            n = len(actions)
            prot = protected_mask(actions, a.protect)
            limit = first_close(actions) - a.before_grasp if a.mode == "approach" else n
            mask = np.ones(n, dtype=bool)
            new_actions = actions.copy()
            for s, e in idle_runs(actions, a.threshold):
                if e - s <= a.keep or prot[s:e].any() or e > limit:
                    continue
                if a.mode == "fill":
                    if s == 0:
                        continue
                    new_actions[s + a.keep : e, :6] = actions[s - 1, :6]
                else:
                    mask[s + a.keep : e] = False
                edited += e - s - a.keep
            g = go.create_group(name)
            cut_group(demo, g, mask, n)
            if a.mode == "fill":
                del g["actions"]
                g.create_dataset("actions", data=new_actions)
            g.attrs["num_samples"] = int(mask.sum())
            tot_in += n
            tot_out += int(mask.sum())
        go.attrs["total"] = tot_out
        go.attrs["edit_idle"] = (
            f"mode={a.mode} keep={a.keep} protect={a.protect} threshold={a.threshold}"
            f" before_grasp={a.before_grasp} src={a.src}"
        )
    print(f"{a.mode}: {tot_in} -> {tot_out} steps, {edited} idle steps edited ({edited / tot_in:.0%})")


if __name__ == "__main__":
    main()
